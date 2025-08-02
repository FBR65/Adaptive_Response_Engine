from qdrant_client import QdrantClient, models
from tika import parser
from sklearn.metrics.pairwise import cosine_similarity
import logging
import secrets
import uuid
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from openai import OpenAI
from typing import List, Dict, Optional

# Environment variables mit Fallbacks
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "manhattan")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("API_KEY", "your-api-key")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
SPACY_MODEL = os.getenv("SPACY_MODEL", "de_core_news_lg")
SEMANTIC_CHUNK_MODEL = os.getenv("SEMANTIC_CHUNK_MODEL", "bge-m3:latest")
TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL", "http://localhost:9998/tika")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globale Variablen für spaCy (einmal laden)
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    logger.warning(f"SpaCy Modell {SPACY_MODEL} nicht gefunden, verwende Fallback")
    nlp = None

# SentenceTransformer wird nicht mehr benötigt - Embeddings über Ollama
logger.info("Verwende Ollama für Embeddings statt lokales SentenceTransformer")


class Document:
    """
    Strukturierte Ausgabe für das geparste Dokument.
    """

    def __init__(self, inhalt: str, metadaten: dict = None):
        self.inhalt = inhalt
        self.metadaten = metadaten if metadaten is not None else {}


def semantic_chunking(text, threshold_percentile=25):
    """
    Semantisches Chunking mit Fallback für fehlende Modelle
    Da Embeddings über Ollama gemacht werden, verwenden wir einfaches Satz-basiertes Chunking
    """
    if nlp is None:
        # Fallback: Einfaches Satz-basiertes Chunking
        sentences = text.split(". ")
        # Gruppiere Sätze in Chunks von ca. 3-5 Sätzen
        chunks = []
        current_chunk = []
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence.strip())
            if len(current_chunk) >= 4 or i == len(sentences) - 1:
                chunks.append(". ".join(current_chunk))
                current_chunk = []
        return chunks

    # Schritt 1: Sätze aufteilen mit spaCy
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Da kein lokales Embedding-Modell verfügbar ist, verwende einfache Längen-basierte Chunking
    chunks, current_chunk = [], []
    current_length = 0
    target_chunk_length = len(text) // max(4, len(sentences) // 4)  # Ziel: ~4 Chunks

    for sentence in sentences:
        current_chunk.append(sentence.text)
        current_length += len(sentence.text)

        if current_length >= target_chunk_length and len(current_chunk) >= 2:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class IntegratedKnowledgeSystem:
    def __init__(self, collection_name):
        """
        Initialisiert das IntegratedKnowledgeSystem mit dem Namen der Qdrant-Collection.
        """
        self.collection_name = f"{collection_name}_{DISTANCE_METRIC.upper()}"
        self.openai_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=QDRANT_TIMEOUT,
        )
        self._create_collection()

    def _create_collection(self):
        """
        Erstellt die Qdrant-Collection für dense vectors und Volltextsuche.
        """
        if not self.qdrant_client.collection_exists(
            collection_name=self.collection_name
        ):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.MANHATTAN,
                ),
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            logging.info(f"Collection '{self.collection_name}' created.")
        else:
            logging.info(f"Collection '{self.collection_name}' already exists.")

    def generate_point_id(self):
        """
        Generiert eine eindeutige Point-ID.
        """
        uuid_value = uuid.uuid4().hex
        modified_uuid = "".join(
            (
                hex((int(c, 16) ^ secrets.randbits(4) & 15 >> int(c) // 4))[2:]
                if c in "018"
                else c
            )
            for c in uuid_value
        )
        logging.info(f"Created point id '{modified_uuid}'.")
        return str(modified_uuid)

    def stream_document(self, file_path):
        """
        Parsen des Dokuments mit Tika.
        """
        logging.info(f"Streaming document: {file_path}")
        parsed = parser.from_file(file_path, serverEndpoint=TIKA_SERVER_URL)
        metadata = parsed.get("metadata", {})
        if "resourceName" in metadata:
            resource_name = metadata["resourceName"]
            if isinstance(resource_name, list):
                metadata["file_name"] = resource_name[0].strip("b'")
            else:
                metadata["file_name"] = resource_name.strip("b'")
            del metadata["resourceName"]
        content = parsed.get("content", "")
        return Document(inhalt=content, metadaten=metadata)

    def split_into_chunks(self, text, output_dir="temp_chunks"):
        """
        Teilt den Text in semantische Chunks auf.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        chunk_files = []
        chunks = semantic_chunking(text)
        for i, chunk in enumerate(chunks):
            chunk_filename = os.path.join(output_dir, f"chunk_{i}.json")
            with open(chunk_filename, "w", encoding="utf-8") as f:
                json.dump({"text": chunk}, f)
            chunk_files.append(chunk_filename)
        logging.info(f"Saved {len(chunk_files)} chunks to '{output_dir}'.")
        return chunk_files

    def index_document(self, file_path):
        """
        Verarbeitet ein Dokument, teilt es in Chunks auf und speichert diese in der Qdrant-Datenbank.
        """
        logging.info(f"Indexing document: {file_path}")
        document = self.stream_document(file_path)
        chunk_files = self.split_into_chunks(document.inhalt)
        self._fill_database(chunk_files)
        # Bereinigung der temporären Chunk-Dateien
        for file_path in chunk_files:
            os.remove(file_path)
        if os.path.exists("temp_chunks") and not os.listdir("temp_chunks"):
            os.rmdir("temp_chunks")
        logging.info(f"Document '{os.path.basename(file_path)}' indexed successfully.")

    def _fill_database(self, chunk_file_paths):
        """
        Liest Chunk-Dateien, erstellt Embeddings und speichert sie in Qdrant.
        """
        points = []
        for file_path in chunk_file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    chunk = chunk_data["text"]

                response = self.openai_client.embeddings.create(
                    input=[chunk], model=os.getenv("EMBEDDING_MODEL")
                )
                dense_embedding = response.data[0].embedding

                point_id = self.generate_point_id()
                payload = {"text": chunk, "chunk_id": point_id}
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=dense_embedding,
                        payload=payload,
                    )
                )

            except Exception as e:
                logging.error(f"Error processing chunk file '{file_path}': {e}")

        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
            logging.info(
                f"Successfully uploaded {len(points)} chunks to '{self.collection_name}'."
            )
        else:
            logging.info("No data to upload.")

    def query_knowledge(
        self, query: str, limit: int = 10, metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Führt eine Suche in der Qdrant-Datenbank durch mit dense vectors und Volltextsuche.

        Args:
            query (str): Die Suchanfrage.
            limit (int, optional): Die maximale Anzahl der zurückzugebenden Ergebnisse. Defaults to 10.
            metadata_filter (Dict, optional): Ein optionaler Filter basierend auf Metadaten. Defaults to None.

        Returns:
            List[Dict]: Eine Liste der Payloads der relevantesten Knowledge-Einheiten.
        """
        response = self.openai_client.embeddings.create(
            input=[query], model=EMBEDDING_MODEL
        )
        dense_query = response.data[0].embedding

        search_params = models.SearchParams(hnsw_ef=128)

        try:
            must_conditions = []
            if metadata_filter:
                must_conditions.extend(
                    [
                        {"key": key, "match": {"value": value}}
                        for key, value in metadata_filter.items()
                    ]
                )

            # Add text search condition
            must_conditions.append({"key": "text", "match": {"text": query}})

            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=dense_query,
                limit=limit,
                query_filter=models.Filter(must=must_conditions)
                if must_conditions
                else None,
                with_payload=True,
                search_params=search_params,
                score_threshold=0.0,
            )
            return [hit.payload for hit in search_result]
        except Exception as e:
            logging.error(f"Error during query: {e}")
            return []

    def get_specific_knowledge(self, point_ids: List[str]) -> List[Dict]:
        """
        Ruft spezifische Knowledge-Einheiten anhand ihrer IDs ab.

        Args:
            point_ids (List[str]): Eine Liste der IDs der abzurufenden Knowledge-Einheiten.

        Returns:
            List[Dict]: Eine Liste der Payloads der abgerufenen Knowledge-Einheiten.
        """
        points = self.qdrant_client.get_points(
            collection_name=self.collection_name, ids=point_ids, with_payload=True
        )
        return [point.payload for point in points]


if __name__ == "__main__":
    collection_name = "integrated_knowledge_test"
    knowledge_system = IntegratedKnowledgeSystem(collection_name=collection_name)

    # Pfad zu Ihrer Testdatei
    file_path = "./data/datenstrategie.pdf"
    if os.path.exists(file_path):
        knowledge_system.index_document(file_path)
    else:
        print(f"Datei nicht gefunden: {file_path}")

    # Beispielhafte Suchanfrage
    query = "Welche Vorteile hat die Datenstrategie?"
    results = knowledge_system.query_knowledge(query, limit=5)
    print("\nSuchergebnisse:")
    for result in results:
        print(f"  {result.get('text', 'Kein Text')}")

    # Beispielhaftes Abrufen spezifischer IDs (ggf. aus vorherigen Suchergebnissen)
    # specific_ids = ["...", "..."]
    # specific_knowledge = knowledge_system.get_specific_knowledge(specific_ids)
    # print("\nSpezifisches Wissen:")
    # for item in specific_knowledge:
    #     print(f"  {item.get('text', 'Kein Text')}")
