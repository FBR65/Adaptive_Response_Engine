from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class DuckDuckGoSearchResult(BaseModel):
    """
    Represents a single search result from DuckDuckGo.
    """

    title: str = Field(..., description="The title of the search result.")
    link: str = Field(..., description="The URL of the search result.")
    snippet: str = Field(
        ..., description="A short description or snippet of the search result."
    )


class DuckDuckGoSearchResults(BaseModel):
    """
    Represents a collection of search results from DuckDuckGo.
    """

    results: List[DuckDuckGoSearchResult] = Field(
        ..., description="A list of DuckDuckGo search results."
    )


class DuckDuckGoSearcher:
    """
    OPTIMIERTE DuckDuckGo-Suche mit neuer DDGS v9.5+ API.
    Nutzt automatische Backend-Fallbacks und robuste Fehlerbehandlung.
    """

    def __init__(self, proxy: Optional[str] = None, timeout: int = 10):
        """
        Initialisiert den DuckDuckGo-Searcher.

        Args:
            proxy: Optional proxy string (z.B. "socks5h://127.0.0.1:9150")
            timeout: Timeout für Requests in Sekunden
        """
        self.proxy = proxy
        self.timeout = timeout
        self.ddgs = None
        logger.info(
            f"DuckDuckGoSearcher initialisiert - Proxy: {proxy}, Timeout: {timeout}s"
        )

    def search(
        self, query: str, max_results: int = 10, region: str = "de-de"
    ) -> DuckDuckGoSearchResults:
        """
        VERBESSERTE Suche mit der neuen DDGS API v9.5+.

        Args:
            query: The search query string.
            max_results: Maximum number of search results to return (default: 10).
            region: Search region (default: "de-de" for German results).

        Returns:
            A DuckDuckGoSearchResults object containing search results.
        """
        try:
            logger.info(
                f"DDGS-Suche: '{query}' (max_results={max_results}, region={region})"
            )

            # Initialisiere DDGS mit Proxy und Timeout
            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout)

            # Neue API: Direkte text()-Methode mit Backend-Fallbacks
            raw_results = ddgs.text(
                query=query,
                region=region,
                safesearch="moderate",
                timelimit=None,  # Keine zeitliche Begrenzung
                max_results=max_results,
                page=1,
                backend="auto",  # Automatische Backend-Auswahl mit Fallbacks
            )

            logger.info(f"DDGS lieferte {len(raw_results)} Rohergebnisse")

            # Konvertiere die Ergebnisse ins erwartete Format
            formatted_results = []
            for i, result in enumerate(raw_results):
                try:
                    formatted_result = DuckDuckGoSearchResult(
                        title=result.get("title", f"Ergebnis {i + 1}"),
                        link=result.get(
                            "href", ""
                        ),  # Neue API nutzt "href" statt "link"
                        snippet=result.get(
                            "body", "Keine Beschreibung verfügbar"
                        ),  # Neue API nutzt "body" statt "snippet"
                    )
                    formatted_results.append(formatted_result)
                    logger.debug(f"Result {i + 1}: {formatted_result.title[:50]}...")

                except Exception as e:
                    logger.warning(f"Fehler beim Formatieren von Ergebnis {i + 1}: {e}")
                    continue

            logger.info(
                f"DDGS-Suche erfolgreich: {len(formatted_results)} Ergebnisse formatiert"
            )
            return DuckDuckGoSearchResults(results=formatted_results)

        except RatelimitException as e:
            logger.error(f"DDGS Rate-Limit erreicht für Query '{query}': {e}")
            return DuckDuckGoSearchResults(results=[])

        except TimeoutException as e:
            logger.error(f"DDGS Timeout für Query '{query}': {e}")
            return DuckDuckGoSearchResults(results=[])

        except DDGSException as e:
            logger.error(f"DDGS-spezifischer Fehler für Query '{query}': {e}")
            return DuckDuckGoSearchResults(results=[])

        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei DDGS-Suche für Query '{query}': {e}")
            return DuckDuckGoSearchResults(results=[])

    def search_with_backend(
        self,
        query: str,
        max_results: int = 10,
        backend: str = "duckduckgo,google,brave",
        region: str = "de-de",
    ) -> DuckDuckGoSearchResults:
        """
        Erweiterte Suche mit spezifischen Backends.

        Args:
            query: Suchanfrage
            max_results: Maximale Anzahl Ergebnisse
            backend: Komma-getrennte Liste von Backends (z.B. "duckduckgo,google,brave")
            region: Suchregion

        Returns:
            DuckDuckGoSearchResults mit Suchergebnissen
        """
        try:
            logger.info(f"DDGS-Suche mit Backends '{backend}': '{query}'")

            ddgs = DDGS(proxy=self.proxy, timeout=self.timeout)

            raw_results = ddgs.text(
                query=query,
                region=region,
                safesearch="moderate",
                max_results=max_results,
                backend=backend,  # Spezifische Backend-Reihenfolge
            )

            formatted_results = []
            for i, result in enumerate(raw_results):
                try:
                    formatted_result = DuckDuckGoSearchResult(
                        title=result.get("title", f"Ergebnis {i + 1}"),
                        link=result.get("href", ""),
                        snippet=result.get("body", "Keine Beschreibung verfügbar"),
                    )
                    formatted_results.append(formatted_result)
                except Exception as e:
                    logger.warning(f"Fehler beim Formatieren: {e}")
                    continue

            logger.info(
                f"Backend-Suche erfolgreich: {len(formatted_results)} Ergebnisse"
            )
            return DuckDuckGoSearchResults(results=formatted_results)

        except Exception as e:
            logger.error(f"Fehler bei Backend-Suche: {e}")
            return DuckDuckGoSearchResults(results=[])


# Legacy-Unterstützung: Wrapper für alte API
class DuckDuckGoSearcher_Legacy(DuckDuckGoSearcher):
    """Legacy-Wrapper für alte API-Kompatibilität."""

    def search(self, query: str, num_results: int = 10) -> DuckDuckGoSearchResults:
        """Legacy-Methode mit alter Parameterbenennung."""
        logger.info(
            f"Legacy-API-Aufruf: num_results={num_results} → max_results={num_results}"
        )
        return super().search(query, max_results=num_results)


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def test_ddgs_search():
        """Test der optimierten DDGS-Suche."""

        # Standard-Suche
        searcher = DuckDuckGoSearcher()

        test_queries = [
            "KI Medizin Vorteile",
            "Was sind die Vorteile von KI in der Medizin",
            "artificial intelligence healthcare benefits",
        ]

        for query in test_queries:
            print(f"\n🔍 Teste Query: '{query}'")
            print("=" * 60)

            try:
                # Standard-Suche
                results = searcher.search(query, max_results=5)

                if results.results:
                    print(f"✅ {len(results.results)} Ergebnisse gefunden:")
                    for i, result in enumerate(results.results, 1):
                        print(f"\n{i}. {result.title}")
                        print(f"   URL: {result.link}")
                        print(f"   Snippet: {result.snippet[:100]}...")
                else:
                    print("❌ Keine Ergebnisse gefunden")

            except Exception as e:
                print(f"❌ Fehler bei der Suche: {e}")

        # Test mit verschiedenen Backends
        print("\n🚀 Test mit spezifischen Backends:")
        print("=" * 60)

        backend_results = searcher.search_with_backend(
            "KI Medizin Vorteile", max_results=3, backend="duckduckgo,google,brave"
        )

        if backend_results.results:
            print(f"✅ Backend-Suche: {len(backend_results.results)} Ergebnisse")
            for i, result in enumerate(backend_results.results, 1):
                print(f"{i}. {result.title[:50]}...")
        else:
            print("❌ Backend-Suche fehlgeschlagen")

    # Führe Tests aus
    asyncio.run(test_ddgs_search())
