"""
Response Generation Agent - Generiert Antworten basierend auf verfügbaren Informationsquellen
"""

import logging
from typing import Dict, Any, List, Optional
import requests
import json
import os

logger = logging.getLogger(__name__)


class ResponseGenerationAgent:
    """
    Agent zur Generierung von Antworten basierend auf verfügbaren Informationsquellen.
    Nutzt RAG, Web-Suche und andere Services zur Informationsbeschaffung.
    """

    def __init__(self):
        """Initialisiert den Response Generation Agent mit OpenAI-kompatibler API."""
        # OpenAI-kompatible API Konfiguration (ollama, vllm, etc.)
        self.api_base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv(
            "OPENAI_API_KEY", "ollama"
        )  # ollama braucht keinen echten key
        self.model = os.getenv("OPENAI_MODEL", "qwen2.5:latest")

        logger.info(
            f"ResponseGenerationAgent - API: {self.api_base}, Model: {self.model}"
        )

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
                timeout=60,  # Erhöht von 30s auf 60s für komplexe Anfragen
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

    async def generate_response(
        self,
        query: str,
        analysis: Dict[str, Any],
        mcp_tools: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,  # Fange alle unerwarteten Parameter ab
    ) -> Dict[str, Any]:
        """
        Generiert eine Antwort basierend auf der Query-Analyse und verfügbaren Tools.

        Args:
            query: Die (verfeinerte) Nutzereingabe
            analysis: Analyseergebnis vom Query Analysis Agent
            mcp_tools: Verfügbare MCP-Tools
            context: Zusätzlicher Kontext
            **kwargs: Fange unerwartete Parameter ab und logge sie

        Returns:
            Dictionary mit generierter Antwort und Metadaten
        """
        # Debug: Logge unerwartete Parameter
        if kwargs:
            logger.warning(
                f"⚠️ Unerwartete Parameter in generate_response: {list(kwargs.keys())}"
            )
            for key, value in kwargs.items():
                logger.warning(f"  - {key}: {value}")

        try:
            logger.info(f"Generiere Antwort für Query: {query[:100]}...")

            # Sammle Informationen aus verschiedenen Quellen
            gathered_info = await self._gather_information(
                query, analysis, mcp_tools, context
            )

            # Generiere Antwort basierend auf gesammelten Informationen
            response = await self._synthesize_response(query, analysis, gathered_info)

            return {
                "response": response,
                "sources_used": gathered_info.get("sources_used", []),
                "confidence": gathered_info.get("confidence", 0.7),
                "information_completeness": gathered_info.get("completeness", 0.5),
                "processing_time": gathered_info.get("processing_time", 0),
                "metadata": {
                    "query": query,
                    "analysis": analysis,
                    "sources": gathered_info.get("raw_sources", {}),
                },
            }

        except Exception as e:
            logger.error(f"Fehler bei der Antwortgenerierung: {e}")
            # Fallback-Antwort als Dictionary zurückgeben
            fallback_response = f"Entschuldigung, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten: {str(e)}"
            return {
                "response": fallback_response,
                "sources_used": [],
                "confidence": 0.1,
                "information_completeness": 0.0,
                "processing_time": 0,
                "metadata": {
                    "query": query,
                    "analysis": analysis,
                    "error": str(e),
                    "fallback": True,
                },
            }

    async def _gather_information(
        self,
        query: str,
        analysis: Dict[str, Any],
        mcp_tools: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        OPTIMIERTE Informationssammlung für 95%+ Qualitätsstandard.

        Args:
            query: Die Suchanfrage
            analysis: Query-Analyse-Ergebnis
            mcp_tools: Verfügbare MCP-Tools
            context: Erweiterte Kontext-Informationen für Qualitätsoptimierung

        Returns:
            Dictionary mit hochwertigen gesammelten Informationen
        """
        import time

        start_time = time.time()

        gathered_info = {
            "sources_used": [],
            "raw_sources": {},
            "confidence": 0.0,
            "completeness": 0.0,
        }

        required_sources = analysis.get("required_sources", ["rag"])

        # SICHERSTELLEN, dass RAG IMMER verwendet wird
        if "rag" not in required_sources:
            required_sources.append("rag")

        # QUALITÄTS-MODUS: Erweiterte Suchstrategie
        if context and context.get("quality_focus", False):
            logger.info("QUALITÄTS-MODUS aktiviert - Erweiterte Informationssammlung")

            # Erweiterte Quellen für höhere Qualität
            if "web_search" not in required_sources:
                required_sources.append("web_search")
            if context.get("comprehensive_search", False):
                required_sources.extend(["web_content", "time"])

        # RAG-System abfragen - VERBESSERT
        if "rag" in required_sources:
            rag_info = await self._query_rag_system_enhanced(query, mcp_tools, context)
            if rag_info:
                gathered_info["sources_used"].append("rag")
                gathered_info["raw_sources"]["rag"] = rag_info
                gathered_info["confidence"] += 0.4  # Erhöht von 0.3

        # Web-Suche durchführen - VERBESSERT
        if "web_search" in required_sources:
            web_info = await self._query_web_search_enhanced(query, mcp_tools, context)
            if web_info:
                gathered_info["sources_used"].append("web_search")
                gathered_info["raw_sources"]["web_search"] = web_info
                gathered_info["confidence"] += 0.4  # Bleibt bei 0.4

        # ZUSÄTZLICHE Web-Suche bei Qualitätsfokus
        if context and context.get("intensive_search", False):
            additional_web_info = await self._query_additional_web_sources(
                query, mcp_tools
            )
            if additional_web_info:
                gathered_info["sources_used"].append("additional_web")
                gathered_info["raw_sources"]["additional_web"] = additional_web_info
                gathered_info["confidence"] += 0.2

        # Zeitinformationen abrufen
        if "time" in required_sources:
            time_info = await self._get_current_time(mcp_tools)
            if time_info:
                gathered_info["sources_used"].append("time")
                gathered_info["raw_sources"]["time"] = time_info
                gathered_info["confidence"] += 0.1

        # Website-Inhalte extrahieren (falls URLs in der Query) - VERBESSERT
        if (
            self._contains_urls(query)
            or context
            and context.get("web_content_extraction", False)
        ):
            web_content = await self._extract_web_content_enhanced(
                query, mcp_tools, context
            )
            if web_content:
                gathered_info["sources_used"].append("web_content")
                gathered_info["raw_sources"]["web_content"] = web_content
                gathered_info["confidence"] += 0.2

        # SCHÄRFERE Vollständigkeits-Berechnung
        total_sources = len(required_sources)
        obtained_sources = len(gathered_info["sources_used"])

        # Bei Qualitätsmodus: Höhere Anforderungen
        if context and context.get("quality_focus", False):
            # Mindestens 80% der Quellen müssen verfügbar sein für gute Vollständigkeit
            gathered_info["completeness"] = (
                min(obtained_sources / total_sources, 1.0) if total_sources > 0 else 0.3
            )
            if gathered_info["completeness"] < 0.8:
                logger.warning(
                    f"Niedrige Vollständigkeit im Qualitätsmodus: {gathered_info['completeness']:.1%}"
                )
        else:
            gathered_info["completeness"] = (
                min(obtained_sources / total_sources, 1.0) if total_sources > 0 else 0.5
            )

        gathered_info["processing_time"] = time.time() - start_time
        gathered_info["confidence"] = min(gathered_info["confidence"], 1.0)

        logger.info(
            f"Informationen gesammelt: {gathered_info['sources_used']}, Vollständigkeit: {gathered_info['completeness']:.2f}"
        )

        return gathered_info

    async def _query_rag_system(
        self, query: str, mcp_tools: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Fragt das RAG-System ab."""
        try:
            if not mcp_tools or "query_knowledge" not in mcp_tools:
                logger.warning("RAG-System nicht verfügbar")
                return None

            # Simuliere MCP-Tool-Aufruf
            # In der echten Implementierung würde hier der MCP-Client verwendet
            result = await mcp_tools["query_knowledge"](query=query, limit=5)

            if result and result.get("results"):
                return {
                    "results": result["results"],
                    "query": query,
                    "source": "rag_system",
                }

            return None

        except Exception as e:
            logger.error(f"Fehler beim RAG-System-Aufruf: {e}")
            return None

    async def _query_web_search(
        self, query: str, mcp_tools: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Führt eine Web-Suche durch."""
        try:
            if not mcp_tools or "duckduckgo_search" not in mcp_tools:
                logger.warning("Web-Suche nicht verfügbar")
                return None

            # Simuliere MCP-Tool-Aufruf
            result = await mcp_tools["duckduckgo_search"](query=query, max_results=5)

            if result and result.get("results"):
                return {
                    "results": result["results"],
                    "query": query,
                    "source": "web_search",
                }

            return None

        except Exception as e:
            logger.error(f"Fehler bei der Web-Suche: {e}")
            return None

    async def _get_current_time(
        self, mcp_tools: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Ruft aktuelle Zeitinformationen ab."""
        try:
            if not mcp_tools or "get_current_time_utc" not in mcp_tools:
                logger.warning("Zeit-Service nicht verfügbar")
                return None

            # Simuliere MCP-Tool-Aufruf
            result = await mcp_tools["get_current_time_utc"]()

            if result and result.get("current_time_utc"):
                return {
                    "current_time": result["current_time_utc"],
                    "source": "ntp_time",
                }

            return None

        except Exception as e:
            logger.error(f"Fehler beim Zeit-Service: {e}")
            return None

    async def _extract_web_content(
        self, query: str, mcp_tools: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Extrahiert Inhalte von Websites."""
        try:
            urls = self._extract_urls_from_query(query)
            if not urls or not mcp_tools or "extract_website_text" not in mcp_tools:
                return None

            extracted_content = []
            for url in urls[:3]:  # Limitiere auf 3 URLs
                try:
                    result = await mcp_tools["extract_website_text"](url=url)
                    if result and result.get("text_content"):
                        extracted_content.append(
                            {
                                "url": url,
                                "content": result["text_content"][
                                    :2000
                                ],  # Limitiere Inhalt
                                "success": True,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Fehler beim Extrahieren von {url}: {e}")
                    extracted_content.append(
                        {"url": url, "error": str(e), "success": False}
                    )

            return (
                {
                    "extracted_pages": extracted_content,
                    "source": "web_content_extraction",
                }
                if extracted_content
                else None
            )

        except Exception as e:
            logger.error(f"Fehler bei der Web-Content-Extraktion: {e}")
            return None

    def _contains_urls(self, text: str) -> bool:
        """Prüft, ob der Text URLs enthält."""
        import re

        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return bool(re.search(url_pattern, text))

    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extrahiert URLs aus einer Query."""
        import re

        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return re.findall(url_pattern, query)

    async def _synthesize_response(
        self, query: str, analysis: Dict[str, Any], gathered_info: Dict[str, Any]
    ) -> str:
        """
        Synthetisiert eine Antwort aus den gesammelten Informationen.

        Args:
            query: Die ursprüngliche Anfrage
            analysis: Query-Analyse
            gathered_info: Gesammelte Informationen

        Returns:
            Synthetisierte Antwort
        """
        try:
            # Erstelle Kontext aus gesammelten Informationen
            context_parts = []

            # RAG-Informationen
            if "rag" in gathered_info.get("raw_sources", {}):
                rag_data = gathered_info["raw_sources"]["rag"]
                if rag_data and rag_data.get("results"):
                    context_parts.append("Wissensbasis-Informationen:")
                    for i, item in enumerate(rag_data["results"][:3], 1):
                        if isinstance(item, dict) and "text" in item:
                            context_parts.append(f"{i}. {item['text'][:500]}...")

            # Web-Such-Informationen
            if "web_search" in gathered_info.get("raw_sources", {}):
                web_data = gathered_info["raw_sources"]["web_search"]
                if web_data and web_data.get("results"):
                    context_parts.append("\nWeb-Such-Ergebnisse:")
                    for i, item in enumerate(web_data["results"][:3], 1):
                        title = item.get("title", "Unbekannt")
                        snippet = item.get("snippet", "")
                        url = item.get(
                            "link", item.get("href", "")
                        )  # KORREKTUR: URL hinzufügen
                        context_parts.append(f"{i}. {title}: {snippet}")
                        if url:  # URL separat hinzufügen für bessere Verfügbarkeit
                            context_parts.append(f"   URL: {url}")

            # Zusätzliche Web-Quellen
            if "additional_web" in gathered_info.get("raw_sources", {}):
                additional_web_data = gathered_info["raw_sources"]["additional_web"]
                if additional_web_data and additional_web_data.get("results"):
                    context_parts.append("\nZusätzliche Web-Ergebnisse:")
                    for i, item in enumerate(additional_web_data["results"][:3], 1):
                        title = item.get("title", "Unbekannt")
                        snippet = item.get("snippet", "")
                        url = item.get("link", item.get("href", ""))
                        context_parts.append(f"{i}. {title}: {snippet}")
                        if url:
                            context_parts.append(f"   URL: {url}")

            # Zeit-Informationen
            if "time" in gathered_info.get("raw_sources", {}):
                time_data = gathered_info["raw_sources"]["time"]
                if time_data:
                    context_parts.append(
                        f"\nAktuelle Zeit: {time_data.get('current_time', 'Unbekannt')}"
                    )

            # Web-Content
            if "web_content" in gathered_info.get("raw_sources", {}):
                web_content = gathered_info["raw_sources"]["web_content"]
                if web_content and web_content.get("extracted_pages"):
                    context_parts.append("\nExtrahierte Web-Inhalte:")
                    for page in web_content["extracted_pages"]:
                        if page.get("success"):
                            context_parts.append(
                                f"- {page['url']}: {page['content'][:300]}..."
                            )

            context_text = (
                "\n".join(context_parts)
                if context_parts
                else "Keine zusätzlichen Informationen verfügbar."
            )

            # Erstelle qualitätsorientiertes Antwort-Prompt
            response_prompt = f"""
            Erstelle eine hochwertige, strukturierte Antwort für folgende Frage:

            Frage: "{query}"

            Verfügbare Informationen:
            {context_text}

            ULTRA-STRENGE QUALITÄTS-ANFORDERUNGEN für 95%+ Standard:
            1. **Vollständigkeit**: Beantworte ALLE Aspekte der Frage umfassend
            2. **Struktur**: Verwende klare Überschriften (###) und Aufzählungen
            3. **Beispiele**: Integriere mindestens 3-5 konkrete, spezifische Beispiele
            4. **ECHTE URLs**: Verwende NUR die oben bereitgestellten URLs! NIEMALS [URL_1], [URL_2] Platzhalter!
            5. **Präzision**: Jede Aussage muss faktisch korrekt und wissenschaftlich fundiert sein
            6. **Relevanz**: Jeder Absatz muss direkt zur Beantwortung der Frage beitragen
            7. **Kohärenz**: Logischer Aufbau mit fließenden Übergängen
            8. **Tiefe**: Erkläre Zusammenhänge und Hintergründe detailliert

            KRITISCH - URL-VERWENDUNG:
            - Nutze AUSSCHLIESSLICH die oben angegebenen URLs aus den Web-Suchergebnissen
            - Format: [Beschreibender Text](echte-url-hier)
            - NIEMALS Platzhalter wie [URL_1], [URL_2] etc. verwenden!
            - Wenn keine URL verfügbar ist, dann keine URL angeben
            
            WICHTIG: 
            - Falls Informationen unvollständig sind, ergänze basierend auf Fachwissen
            - Strukturiere mit ### Überschriften für bessere Lesbarkeit
            - Verwende **Fettschrift** für wichtige Punkte
            - Füge praktische Handlungsempfehlungen hinzu

            Erstelle jetzt eine exzellente, wissenschaftlich fundierte Antwort:
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein hilfreicher Assistent, der präzise und strukturierte Antworten gibt.",
                    },
                    {"role": "user", "content": response_prompt},
                ],
                max_tokens=1500,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            generated_response = response_content.strip()

            # Füge detaillierte Quellen-Information hinzu
            if gathered_info.get("sources_used"):
                sources_parts = []

                # Sammle URLs aus Web-Suchen
                if "web_search" in gathered_info.get("raw_sources", {}):
                    web_data = gathered_info["raw_sources"]["web_search"]
                    if web_data and web_data.get("results"):
                        for item in web_data["results"][:2]:  # Nur top 2 URLs
                            url = item.get("link", item.get("href", ""))
                            if url:
                                sources_parts.append(url)

                # Sammle URLs aus zusätzlichen Web-Suchen
                if "additional_web" in gathered_info.get("raw_sources", {}):
                    additional_web_data = gathered_info["raw_sources"]["additional_web"]
                    if additional_web_data and additional_web_data.get("results"):
                        for item in additional_web_data["results"][
                            :2
                        ]:  # Nur top 2 URLs
                            url = item.get("link", item.get("href", ""))
                            if url:
                                sources_parts.append(url)

                # Fallback zu generischen Namen wenn keine URLs
                if not sources_parts:
                    sources_parts = gathered_info["sources_used"]

                sources_text = ", ".join(sources_parts)
                generated_response += f"\n\n*Quellen: {sources_text}*"

            logger.info(f"Antwort generiert (Länge: {len(generated_response)} Zeichen)")
            return generated_response

        except Exception as e:
            logger.error(f"Fehler bei der Antwort-Synthese: {e}")
            # Bei Fehlern gib einen einfachen String zurück, nicht ein Dictionary
            fallback_text = f"Entschuldigung, bei der Verarbeitung ist ein Fehler aufgetreten: {str(e)}"
            return fallback_text

    async def _generate_fallback_response(
        self, query: str, error: str
    ) -> Dict[str, Any]:
        """
        Generiert eine Fallback-Antwort bei Fehlern.

        Args:
            query: Die ursprüngliche Anfrage
            error: Fehlermeldung

        Returns:
            Fallback-Antwort-Dictionary
        """
        fallback_response = f"""
        Entschuldigung, ich konnte Ihre Anfrage "{query}" nicht vollständig bearbeiten.

        Es ist ein technischer Fehler aufgetreten. Bitte versuchen Sie es erneut oder formulieren Sie Ihre Frage anders.

        Für einfache Fragen kann ich Ihnen auch ohne externe Informationsquellen helfen.
        """

        return {
            "response": fallback_response,
            "sources_used": [],
            "confidence": 0.1,
            "information_completeness": 0.0,
            "processing_time": 0,
            "error": error,
            "metadata": {"query": query, "fallback": True},
        }

    async def _query_rag_system_enhanced(
        self,
        query: str,
        mcp_tools: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """VERBESSERTE RAG-System-Abfrage für höhere Qualität."""
        try:
            if not mcp_tools or "query_knowledge" not in mcp_tools:
                logger.warning("RAG-System nicht verfügbar")
                return None

            # Erweiterte RAG-Parameter bei Qualitätsfokus
            limit = 10 if context and context.get("comprehensive_search", False) else 5

            result = await mcp_tools["query_knowledge"](query=query, limit=limit)

            if result and result.get("results"):
                return {
                    "results": result["results"],
                    "query": query,
                    "source": "rag_enhanced",
                    "limit_used": limit,
                }

            return None

        except Exception as e:
            logger.error(f"Fehler beim erweiterten RAG-System-Aufruf: {e}")
            return None

    async def _query_web_search_enhanced(
        self,
        query: str,
        mcp_tools: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """VERBESSERTE Web-Suche - verwendet die ursprüngliche Frage mit minimalem Refinement."""
        try:
            if not mcp_tools or "duckduckgo_search" not in mcp_tools:
                logger.warning("Web-Suche nicht verfügbar")
                return None

            # Verwende die ursprüngliche Frage mit nur minimalem Refinement
            refined_query = self._simple_query_refinement(query)
            logger.info(
                f"🎯 Direkte Suche: '{refined_query}' (Original: '{query[:100]}')"
            )

            # Mehr Ergebnisse bei Qualitätsfokus
            max_results = 8 if context and context.get("intensive_search", False) else 5

            result = await mcp_tools["duckduckgo_search"](
                query=refined_query, max_results=max_results
            )

            if result and result.get("results"):
                return {
                    "results": result["results"],
                    "query": refined_query,
                    "original_query": query,
                    "source": "web_search_enhanced",
                    "max_results_used": max_results,
                }

            return None

        except Exception as e:
            logger.error(f"Fehler bei der erweiterten Web-Suche: {e}")
            return None

    async def _query_additional_web_sources(
        self, query: str, mcp_tools: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        VEREINFACHTE zusätzliche Web-Suche - NUR EINE gute Suche statt mehrere schlechte!
        """
        try:
            if not mcp_tools or "duckduckgo_search" not in mcp_tools:
                return None

            # Verwende die ursprüngliche Frage mit minimalem Refinement
            refined_query = self._simple_query_refinement(query)

            logger.info(f"🎯 EINE zusätzliche Suche: '{refined_query}'")

            # NUR EINE gute Suche mit mehr Ergebnissen
            result = await mcp_tools["duckduckgo_search"](
                query=refined_query,
                max_results=8,  # Mehr Ergebnisse in EINER Suche
            )

            if result and result.get("results"):
                return {
                    "results": result["results"],
                    "source": "additional_web_sources",
                    "query_used": refined_query,
                }

            return None

        except Exception as e:
            logger.error(f"Fehler bei zusätzlichen Web-Quellen: {e}")
            return None

    async def _extract_web_content_enhanced(
        self,
        query: str,
        mcp_tools: Optional[Dict] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """VERBESSERTE Website-Extraktion für höhere Qualität."""
        try:
            if not mcp_tools or "extract_website_text" not in mcp_tools:
                logger.warning("Website-Extraktion nicht verfügbar")
                return None

            # Extrahiere URLs aus Query oder nutze Web-Such-Ergebnisse
            urls = self._extract_urls_from_query(query)

            # Falls keine URLs in Query, nutze Top-Ergebnisse der Web-Suche
            if not urls and context and context.get("quality_focus", False):
                # Hole beste URLs aus vorherigen Web-Such-Ergebnissen
                urls = await self._get_top_urls_from_web_search(query, mcp_tools)

            if not urls:
                return None

            extracted_contents = []
            for url in urls[:3]:  # Max 3 URLs für Performance
                try:
                    result = await mcp_tools["extract_website_text"](url=url)
                    if result and result.get("text"):
                        extracted_contents.append(
                            {
                                "url": url,
                                "text": result["text"][
                                    :2000
                                ],  # Begrenzen auf 2000 Zeichen
                                "title": result.get("title", ""),
                            }
                        )
                except Exception:
                    continue

            if extracted_contents:
                return {
                    "extracted_content": extracted_contents,
                    "source": "web_content_enhanced",
                    "urls_processed": len(extracted_contents),
                }

            return None

        except Exception as e:
            logger.error(f"Fehler bei der erweiterten Website-Extraktion: {e}")
            return None

    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extrahiert URLs aus der Query."""
        import re

        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return re.findall(url_pattern, query)

    def _extract_clean_search_terms(self, complex_query: str) -> str:
        """
        Extrahiert die ursprüngliche Frage OHNE sie zu zerstören.
        Die ursprüngliche Frage ist oft die beste Suchanfrage!

        Args:
            complex_query: Komplexe Query mit Verbesserungsanweisungen

        Returns:
            Die ursprüngliche, unveränderte Frage
        """
        import re

        # 1. Bei "Erweiterte Recherche:" - extrahiere NUR die ursprüngliche Frage
        erweiterte_match = re.search(
            r"Erweiterte Recherche:\s*([^\n]+)", complex_query, re.IGNORECASE
        )
        if erweiterte_match:
            original_question = erweiterte_match.group(1).strip()
            logger.info(f"✅ Extrahierte ursprüngliche Frage: '{original_question}'")
            return original_question

        # 2. Andere Muster für ursprüngliche Fragen
        other_patterns = [
            r"Query:\s*([^\n]+)",
            r"Frage:\s*([^\n]+)",
            r"Thema:\s*([^\n]+)",
        ]

        for pattern in other_patterns:
            match = re.search(pattern, complex_query, re.IGNORECASE)
            if match:
                clean_query = match.group(1).strip()
                logger.info(f"✅ Verwende ursprüngliche Frage direkt: '{clean_query}'")
                return clean_query

        # 3. Falls keine Muster gefunden, nimm die ersten 8 Wörter (ohne Prompt-Anweisungen)
        words = complex_query.split()

        # Entferne nur offensichtliche Prompt-Anweisungen
        filter_words = {
            "erweiterte",
            "recherche",
            "qualitäts-anforderungen",
            "mindestens",
            "verifiable",
            "quellen",
            "strukturierte",
            "gliederung",
            "zwischenüberschriften",
            "verbessern",
            "qualitätsverbesserung",
            "konkrete",
            "beispiele",
            "urls",
            "wissenschaftlich",
            "fundierte",
        }

        clean_words = []
        for word in words[:12]:
            if word.lower() not in filter_words and len(word) > 2:
                clean_words.append(word)
                if len(clean_words) >= 8:  # Max 8 Wörter für natürliche Fragen
                    break

        result = " ".join(clean_words) if clean_words else "KI Medizin Vorteile"
        logger.info(f"✅ Fallback bereinigte Frage: '{result}'")
        return result

    def _simple_query_refinement(self, query: str) -> str:
        """
        EINFACHE Query-Verbesserung - nur minimal optimieren ohne zu zerstören!

        Args:
            query: Ursprüngliche Suchanfrage

        Returns:
            Minimal optimierte Suchanfrage
        """
        import re

        # Nur sehr offensichtliche Verbesserungen
        if not query.strip():
            return "KI Medizin Vorteile"

        # Entferne nur Fragezeichen am Ende (macht Suche besser)
        refined = re.sub(r"\?+$", "", query.strip())

        # Nur bei sehr gesprächigen Fragen minimal kürzen
        if len(refined.split()) > 10 and any(
            word in refined.lower()
            for word in ["kannst du", "könntest du", "können sie"]
        ):
            # Entferne nur direkte Anreden
            refined = re.sub(
                r"\b(kannst du|könntest du|können sie)\b",
                "",
                refined,
                flags=re.IGNORECASE,
            )
            refined = re.sub(r"\s+", " ", refined).strip()

        logger.info(f"🔧 Query-Refinement: '{query}' → '{refined}'")
        return refined if refined else query

    async def _get_top_urls_from_web_search(
        self, query: str, mcp_tools: Optional[Dict] = None
    ) -> List[str]:
        """Holt Top URLs aus Web-Suche für Website-Extraktion."""
        try:
            result = await mcp_tools["duckduckgo_search"](query=query, max_results=3)
            if result and result.get("results"):
                return [
                    r.get("url", r.get("href", ""))
                    for r in result["results"]
                    if r.get("url") or r.get("href")
                ]
            return []
        except Exception:
            return []
