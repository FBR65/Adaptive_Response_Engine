"""
Query Analysis Agent - Analysiert und verbessert Nutzereingaben
"""

import logging
import json
import requests
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class QueryAnalysisAgent:
    """
    Agent zur Analyse und Verbesserung von Nutzereingaben.
    Identifiziert Intent, Kontext und benötigte Informationsquellen.
    """

    def __init__(self):
        """Initialisiert den Query Analysis Agent mit OpenAI-kompatibler API."""
        # OpenAI-kompatible API Konfiguration (ollama, vllm, etc.)
        self.api_base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv(
            "OPENAI_API_KEY", "ollama"
        )  # ollama braucht keinen echten key
        self.model = os.getenv("OPENAI_MODEL", "qwen2.5:latest")

        logger.info(f"QueryAnalysisAgent - API: {self.api_base}, Model: {self.model}")

    def _call_openai_compatible_api(
        self, messages: list, max_tokens: int = 1000
    ) -> str:
        """Ruft OpenAI-kompatible API auf (ollama, vllm, etc.)"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"API Call Error: {e}")
            return None

    async def analyze_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analysiert eine Nutzereingabe und bestimmt Intent, Komplexität und benötigte Ressourcen.

        Args:
            query: Die Nutzereingabe
            context: Zusätzlicher Kontext (optional)

        Returns:
            Dictionary mit Analyseergebnissen
        """
        try:
            logger.info(f"Analysiere Query: {query[:100]}...")

            analysis_prompt = f"""
            Analysiere die folgende Nutzereingabe und gib eine strukturierte Antwort im JSON-Format zurück:

            Nutzereingabe: "{query}"
            
            Kontext: {context if context else "Kein zusätzlicher Kontext"}

            Analysiere folgende Aspekte:
            1. Intent: Was möchte der Nutzer wissen/erreichen?
            2. Komplexität: Einfach (1), Mittel (2), Komplex (3)
            3. Informationsquellen: Welche Quellen könnten hilfreich sein? [rag, web_search, time, none]
            4. Sprache: In welcher Sprache soll geantwortet werden?
            5. Spezifität: Ist die Anfrage spezifisch genug oder braucht sie Nachfragen?
            6. Verbesserte Query: Eine optimierte Version der ursprünglichen Anfrage

            Antwortformat:
            {{
                "intent": "string",
                "complexity": 1|2|3,
                "required_sources": ["rag", "web_search", "time"],
                "language": "de|en",
                "specificity_score": 0.1-1.0,
                "needs_clarification": true|false,
                "improved_query": "string",
                "suggested_questions": ["string1", "string2"],
                "confidence": 0.1-1.0
            }}
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für Queryanalyse. Antworte immer im angegebenen JSON-Format.",
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                max_tokens=1000,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            result_text = response_content

            # Parse JSON response
            analysis_result = json.loads(result_text)

            logger.info(
                f"Query-Analyse abgeschlossen: Intent={analysis_result.get('intent', 'unknown')}"
            )
            return analysis_result

        except json.JSONDecodeError as e:
            logger.error(f"Fehler beim Parsen der JSON-Antwort: {e}")
            # Fallback-Analyse
            return self._create_fallback_analysis(query)
        except Exception as e:
            logger.error(f"Fehler bei der Query-Analyse: {e}")
            return self._create_fallback_analysis(query)

    def _create_fallback_analysis(self, query: str) -> Dict[str, Any]:
        """
        Erstellt eine Fallback-Analyse, wenn die LLM-Analyse fehlschlägt.

        Args:
            query: Die ursprüngliche Nutzereingabe

        Returns:
            Basis-Analyseergebnis
        """
        # Einfache heuristische Analyse
        complexity = 1
        if len(query.split()) > 10:
            complexity = 2
        if any(
            word in query.lower()
            for word in ["komplex", "detail", "analyse", "vergleich"]
        ):
            complexity = 3

        # Bestimme benötigte Quellen basierend auf Keywords
        required_sources = []

        # IMMER RAG verwenden für lokales Wissen
        required_sources.append("rag")

        # Web-Suche für aktuelle oder spezielle Themen
        if any(
            word in query.lower()
            for word in ["aktuell", "neuest", "heute", "jetzt", "news", "entwicklung"]
        ):
            required_sources.extend(["web_search", "time"])
        # Auch für allgemeine Fragen Web-Suche hinzufügen für umfassendere Antworten
        elif len(query.split()) > 3:  # Für längere Fragen
            required_sources.append("web_search")

        # Fallback: Mindestens RAG verwenden
        if not required_sources:
            required_sources = ["rag"]

        return {
            "intent": "Information abrufen",
            "complexity": complexity,
            "required_sources": required_sources,
            "language": "de",
            "specificity_score": 0.7,
            "needs_clarification": False,
            "improved_query": query,
            "suggested_questions": [],
            "confidence": 0.6,
        }

    async def refine_query(self, original_query: str, analysis: Dict[str, Any]) -> str:
        """
        Verfeinert eine Query basierend auf der Analyse.

        Args:
            original_query: Die ursprüngliche Anfrage
            analysis: Das Analyseergebnis

        Returns:
            Verfeinerte Query
        """
        try:
            if analysis.get("confidence", 0) > 0.8:
                return analysis.get("improved_query", original_query)

            refinement_prompt = f"""
            Verbessere die folgende Nutzereingabe basierend auf der Analyse:

            Ursprüngliche Anfrage: "{original_query}"
            
            Analyse:
            - Intent: {analysis.get("intent", "unbekannt")}
            - Komplexität: {analysis.get("complexity", 1)}
            - Spezifität: {analysis.get("specificity_score", 0.5)}

            Erstelle eine verbesserte, spezifischere Version der Anfrage, die:
            1. Klarer formuliert ist
            2. Alle relevanten Aspekte abdeckt
            3. Für eine KI-Antwort optimiert ist

            Antwort nur mit der verbesserten Anfrage, ohne zusätzliche Erklärungen.
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für Query-Optimierung.",
                    },
                    {"role": "user", "content": refinement_prompt},
                ],
                max_tokens=200,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            refined_query = response_content.strip()
            logger.info(
                f"Query verfeinert: {original_query[:50]}... -> {refined_query[:50]}..."
            )

            return refined_query

        except Exception as e:
            logger.error(f"Fehler bei der Query-Verfeinerung: {e}")
            return analysis.get("improved_query", original_query)
