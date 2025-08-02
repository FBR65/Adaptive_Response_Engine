import logging
from typing import List, Dict, Any, Optional
from mcp import ClientSession
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """
    Repräsentiert ein einzelnes Suchergebnis
    """

    title: str
    link: str
    snippet: str


class SearchAgent:
    """
    Agent für die Interaktion mit dem Suchsystem (DuckDuckGo mit Selenium)
    """

    def __init__(self, mcp_client: ClientSession):
        self.mcp_client = mcp_client

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Führt eine Internetsuche durch

        Args:
            query: Die Suchanfrage
            max_results: Maximale Anzahl an Ergebnissen

        Returns:
            Liste der Suchergebnisse
        """
        try:
            # Rufe das Search-Service über MCP auf
            logger.info(f"Suche im Internet nach: {query}")

            # Verwende das MCP-Tool für die Internetsuche
            results = await self.mcp_client.tools["duckduckgo_search"].call(
                query=query, max_results=max_results
            )

            # Konvertiere die Ergebnisse in SearchResult-Objekte
            search_results = []
            for result in results.get("results", []):
                search_results.append(
                    SearchResult(
                        title=result.get("title", ""),
                        link=result.get("link", ""),
                        snippet=result.get("snippet", ""),
                    )
                )

            return search_results
        except Exception as e:
            logger.error(f"Fehler bei der Internetsuche: {e}")
            return []

    async def extract_content(self, url: str) -> str:
        """
        Extrahiert den Inhalt einer Webseite

        Args:
            url: Die URL der Webseite

        Returns:
            Der extrahierte Textinhalt der Webseite
        """
        try:
            # Rufe das Website-Service über MCP auf
            logger.info(f"Extrahiere Inhalt von: {url}")

            # Verwende das MCP-Tool für die Inhaltsextraktion
            result = await self.mcp_client.tools["extract_website_text"].call(url=url)

            return result.get("text_content", "")
        except Exception as e:
            logger.error(f"Fehler bei der Inhaltsextraktion: {e}")
            return ""
