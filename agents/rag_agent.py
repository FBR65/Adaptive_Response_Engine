import logging
from typing import List, Dict, Any, Optional
from mcp import ClientSession
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Agent für die Interaktion mit dem RAG-System (Colpali-RAG mit Qdrant)
    """

    def __init__(self, mcp_client: ClientSession):
        self.mcp_client = mcp_client

    async def query_knowledge(
        self, query: str, limit: int = 10, metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Führt eine Suche im Wissensspeicher durch

        Args:
            query: Die Suchanfrage
            limit: Maximale Anzahl an Ergebnissen
            metadata_filter: Optionale Filter für die Metadaten

        Returns:
            Liste der relevanten Wissenseinträge
        """
        try:
            # Rufe das RAG-Service über MCP auf
            logger.info(f"Suche im Wissensspeicher nach: {query}")

            # Verwende das MCP-Tool für die Wissensabfrage
            results = await self.mcp_client.tools["query_knowledge"].call(
                query=query, limit=limit, metadata_filter=metadata_filter
            )

            return results.get("results", [])
        except Exception as e:
            logger.error(f"Fehler bei der Wissenssuche: {e}")
            return []

    async def index_document(self, file_path: str) -> bool:
        """
        Indiziert ein Dokument im Wissensspeicher

        Args:
            file_path: Pfad zur Datei, die indiziert werden soll

        Returns:
            True, wenn die Indizierung erfolgreich war, sonst False
        """
        try:
            # Rufe das RAG-Service über MCP auf
            logger.info(f"Indiziere Dokument: {file_path}")

            # Verwende das MCP-Tool für die Dokumentenindizierung
            result = await self.mcp_client.tools["index_document"].call(
                file_path=file_path
            )

            return result.get("success", False)
        except Exception as e:
            logger.error(f"Fehler bei der Dokumentenindizierung: {e}")
            return False
