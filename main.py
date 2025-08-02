import asyncio
import logging
import os

# from dotenv import load_dotenv  # Entfernt, da nicht installiert
from agents.iteration_controller import IterationController
from agents.query_analysis_agent import QueryAnalysisAgent
from agents.response_generation_agent import ResponseGenerationAgent
from agents.quality_review_agent import QualityReviewAgent
from agents.adaptive_response_engine import AdaptiveResponseEngine

# Echte MCP Services importieren
from mcp_services.mcp_search.duck_search import DuckDuckGoSearcher
from mcp_services.mcp_qdrant.qdrant_serve import IntegratedKnowledgeSystem
from mcp_services.mcp_website.headless_browser import HeadlessBrowserExtractor
from mcp_services.mcp_time.ntp_time import NtpTime

# Load environment variables
# load_dotenv()  # Entfernt, da nicht installiert

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Hauptfunktion des Adaptive Response Engine Systems
    """
    logger.info("Starte Adaptive Response Engine...")

    try:
        # Initialisiere die echten Agents (ohne LLM erst mal)
        query_agent = QueryAnalysisAgent()
        response_agent = ResponseGenerationAgent()
        quality_agent = QualityReviewAgent()

        # Erstelle Iteration Controller
        controller = IterationController(
            query_agent=query_agent,
            response_agent=response_agent,
            quality_agent=quality_agent,
            max_iterations=3,
            quality_threshold=95.0,  # Sehr hohe Schwelle für echte Qualität
        )

        # Initialisiere echte MCP Services
        try:
            # DuckDuckGo Web-Suche
            web_searcher = DuckDuckGoSearcher()

            # Qdrant RAG System
            rag_system = IntegratedKnowledgeSystem("adaptive_response")

            # Headless Browser für Website-Extraktion
            browser_extractor = HeadlessBrowserExtractor()

            # NTP Zeit Service
            time_service = NtpTime()

            logger.info("Echte MCP Services initialisiert")

        except Exception as e:
            logger.error(f"Fehler beim Initialisieren der MCP Services: {e}")
            raise

        # Echte MCP Tools Funktionen
        async def real_web_search(query: str, max_results: int = 5):
            try:
                logger.info(f"Führe echte Web-Suche durch: {query}")
                search_results = web_searcher.search(
                    query, max_results=max_results
                )  # Neue API: max_results
                return {
                    "results": [
                        {
                            "title": result.title,
                            "snippet": result.snippet,
                            "url": result.link,
                        }
                        for result in search_results.results
                    ]
                }
            except Exception as e:
                logger.error(f"Fehler bei Web-Suche: {e}")
                return {"results": []}

        async def real_rag_query(query: str, limit: int = 5):
            try:
                logger.info(f"Führe echte RAG-Suche durch: {query}")
                results = rag_system.query_knowledge(query, limit=limit)
                return {
                    "results": [
                        {
                            "text": result.get("text", ""),
                            "score": result.get("score", 0.0),
                            "source": result.get("source", "knowledge_base"),
                        }
                        for result in results
                    ]
                }
            except Exception as e:
                logger.error(f"Fehler bei RAG-Suche: {e}")
                return {"results": []}

        async def real_website_extraction(url: str):
            try:
                logger.info(f"Extrahiere Website-Inhalt: {url}")
                text_content = browser_extractor.extract_text(url)
                return {
                    "text_content": text_content,
                    "url": url,
                    "success": text_content is not None,
                }
            except Exception as e:
                logger.error(f"Fehler bei Website-Extraktion: {e}")
                return {"text_content": None, "url": url, "success": False}

        async def real_current_time():
            try:
                logger.info("Hole aktuelle Zeit über NTP")
                time_info = time_service.get_time()
                return {
                    "current_time_utc": time_info.get("utc_time"),
                    "local_time": time_info.get("german_time"),
                    "source": "ntp_server",
                }
            except Exception as e:
                logger.error(f"Fehler bei Zeit-Abfrage: {e}")
                return {"current_time_utc": None, "local_time": None, "source": "error"}

        mcp_tools = {
            "duckduckgo_search": real_web_search,
            "query_knowledge": real_rag_query,
            "extract_website_text": real_website_extraction,
            "get_current_time_utc": real_current_time,
        }

        # Beispiel für die Verarbeitung einer Nutzereingabe
        user_input = "Was sind die Vorteile von KI in der Medizin?"
        logger.info(f"Verarbeite Nutzereingabe: {user_input}")

        # Verarbeite die Nutzereingabe und generiere eine Antwort
        result = await controller.process_query(
            query=user_input, context={"user_mode": True}, mcp_tools=mcp_tools
        )

        # Gebe die finale Antwort aus
        logger.info(f"Finale Antwort: {result['response'][:200]}...")
        print(f"✅ Antwort: {result['response']}")
        print(f"⭐ Qualität: {result['quality_score']:.1f}%")
        print(f"🔄 Iterationen: {result['iterations']}")
        print(f"✅ Erfolgreich: {result['final_acceptable']}")

    except Exception as e:
        logger.error(f"Fehler bei der Ausführung: {e}")
        print(f"Fehler: {e}")

    finally:
        logger.info("Adaptive Response Engine beendet.")


if __name__ == "__main__":
    # Führe die Hauptfunktion asynchron aus
    asyncio.run(main())
