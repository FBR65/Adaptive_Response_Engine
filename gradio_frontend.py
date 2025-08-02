import gradio as gr
import asyncio
import logging
from pathlib import Path

# Import der bestehenden Komponenten
from agents.iteration_controller import IterationController
from agents.query_analysis_agent import QueryAnalysisAgent
from agents.response_generation_agent import ResponseGenerationAgent
from agents.quality_review_agent import QualityReviewAgent

# MCP Services
from mcp_services.mcp_search.duck_search import DuckDuckGoSearcher
from mcp_services.mcp_qdrant.qdrant_serve import IntegratedKnowledgeSystem
from mcp_services.mcp_website.headless_browser import HeadlessBrowserExtractor
from mcp_services.mcp_time.ntp_time import NtpTime

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globale Variablen für die Agenten und Services
controller = None
mcp_tools = None
knowledge_system = None


async def initialize_system():
    """Initialisiert das Adaptive Response Engine System"""
    global controller, mcp_tools, knowledge_system

    try:
        logger.info("Initialisiere Adaptive Response Engine...")

        # Agents initialisieren
        query_agent = QueryAnalysisAgent()
        response_agent = ResponseGenerationAgent()
        quality_agent = QualityReviewAgent()

        # Controller erstellen
        controller = IterationController(
            query_agent=query_agent,
            response_agent=response_agent,
            quality_agent=quality_agent,
            max_iterations=3,
            quality_threshold=95.0,
        )

        # MCP Services initialisieren
        web_searcher = DuckDuckGoSearcher()
        knowledge_system = IntegratedKnowledgeSystem("adaptive_response_MANHATTAN")
        website_extractor = HeadlessBrowserExtractor()
        time_service = NtpTime()

        # MCP Tools definieren
        async def real_web_search(query: str, max_results: int = 5):
            try:
                logger.info(f"Führe echte Web-Suche durch: {query}")
                # DuckDuckGo search ist nicht async, daher entfernen wir await
                search_results = web_searcher.search(query, max_results=max_results)

                # Konvertiere das Pydantic-Modell zu einem Dictionary mit results-Schlüssel
                if hasattr(search_results, "results"):
                    return {
                        "results": [
                            result.model_dump() for result in search_results.results
                        ],
                        "query": query,
                        "max_results": max_results,
                    }
                else:
                    return {"results": [], "query": query, "max_results": max_results}
            except Exception as e:
                logger.error(f"Web-Suche fehlgeschlagen: {e}")
                return {"results": [], "query": query, "max_results": max_results}

        async def real_rag_query(query: str, limit: int = 5):
            try:
                logger.info(f"Führe RAG-Abfrage durch: {query}")
                results = knowledge_system.query_knowledge(query, limit=limit)

                # Konvertiere zu dem Format, das der Response Generation Agent erwartet
                if results and isinstance(results, dict) and "results" in results:
                    return results
                elif results and isinstance(results, list):
                    return {"results": results, "query": query, "source": "rag"}
                else:
                    return {"results": [], "query": query, "source": "rag"}
            except Exception as e:
                logger.error(f"RAG-Abfrage fehlgeschlagen: {e}")
                return {"results": [], "query": query, "source": "rag"}

        async def real_website_extraction(url: str):
            try:
                logger.info(f"Extrahiere Website-Text: {url}")
                result = await website_extractor.extract_text_async(url)
                return result
            except Exception as e:
                logger.error(f"Website-Extraktion fehlgeschlagen: {e}")
                return {"text": "", "title": "", "error": str(e)}

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

        logger.info("System erfolgreich initialisiert!")
        return "✅ System erfolgreich initialisiert!"

    except Exception as e:
        logger.error(f"Fehler bei der Initialisierung: {e}")
        return f"❌ Fehler bei der Initialisierung: {e}"


async def process_query_async(user_input: str):
    """Verarbeitet eine Benutzeranfrage asynchron"""
    if not controller or not mcp_tools:
        return "❌ System nicht initialisiert. Bitte warten Sie, bis die Initialisierung abgeschlossen ist."

    try:
        logger.info(f"Verarbeite Anfrage: {user_input}")

        # Verarbeite die Anfrage
        result = await controller.process_query(
            query=user_input, context={"user_mode": True}, mcp_tools=mcp_tools
        )

        # Gebe nur die finale Antwort zurück (ohne Bewertung)
        return result["response"]

    except Exception as e:
        logger.error(f"Fehler bei der Verarbeitung: {e}")
        return f"❌ Fehler bei der Verarbeitung: {e}"


def process_query(user_input: str):
    """Wrapper für die asynchrone Verarbeitung"""
    if not user_input.strip():
        return "❓ Bitte geben Sie eine Frage ein."

    # Führe die asynchrone Funktion aus
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(process_query_async(user_input))
        return result
    finally:
        loop.close()


def upload_document(files):
    """Lädt Dokumente in das Qdrant-System hoch"""
    if not knowledge_system:
        return "❌ System nicht initialisiert. Bitte warten Sie, bis die Initialisierung abgeschlossen ist."

    if not files:
        return "❓ Bitte wählen Sie mindestens eine Datei zum Hochladen aus."

    results = []

    for file in files:
        try:
            # Verarbeite die hochgeladene Datei
            file_path = file.name
            logger.info(f"Verarbeite Datei: {file_path}")

            # Indexiere das Dokument
            knowledge_system.index_document(file_path)

            file_name = Path(file_path).name
            results.append(f"✅ {file_name} erfolgreich hochgeladen und indexiert")

        except Exception as e:
            file_name = (
                Path(file.name).name if hasattr(file, "name") else "Unbekannte Datei"
            )
            error_msg = f"❌ Fehler beim Verarbeiten von {file_name}: {e}"
            logger.error(error_msg)
            results.append(error_msg)

    return "\n".join(results)


def create_interface():
    """Erstellt das Gradio-Interface"""

    # Monochrome Theme
    theme = gr.themes.Monochrome()

    with gr.Blocks(
        theme=theme,
        title="Adaptive Response Engine",
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .chat-message {
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            background: #f8f9fa;
        }
        """,
    ) as interface:
        gr.Markdown("# 🤖 Adaptive Response Engine")
        gr.Markdown(
            "Ein intelligentes System für hochqualitative Antworten mit Web-Suche und Wissensbasis"
        )

        with gr.Tabs():
            # Tab 1: Chat Interface
            with gr.TabItem("💬 Chat", id=0):
                gr.Markdown("### Stellen Sie Ihre Frage")
                gr.Markdown(
                    "Das System durchsucht das Web und die lokale Wissensbasis für die beste Antwort."
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        user_input = gr.Textbox(
                            label="Ihre Frage",
                            placeholder="Was möchten Sie wissen?",
                            lines=3,
                            show_label=True,
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button(
                            "🔍 Fragen", variant="primary", size="lg"
                        )

                with gr.Row():
                    response_output = gr.Textbox(
                        label="Antwort",
                        lines=15,
                        max_lines=25,
                        show_label=True,
                        interactive=False,
                        show_copy_button=True,
                    )

                # Event handlers
                submit_btn.click(
                    fn=process_query,
                    inputs=[user_input],
                    outputs=[response_output],
                    show_progress=True,
                )

                user_input.submit(
                    fn=process_query,
                    inputs=[user_input],
                    outputs=[response_output],
                    show_progress=True,
                )

            # Tab 2: Wissens-Upload
            with gr.TabItem("📚 Wissen hochladen", id=1):
                gr.Markdown("### Dokumente zur Wissensbasis hinzufügen")
                gr.Markdown(
                    "Laden Sie PDF-, TXT-, DOCX- oder andere Dokumente hoch, um die lokale Wissensbasis zu erweitern."
                )

                with gr.Row():
                    file_upload = gr.File(
                        label="Dokumente auswählen",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".docx", ".doc", ".md", ".html"],
                        show_label=True,
                    )

                with gr.Row():
                    upload_btn = gr.Button(
                        "📤 Hochladen und Indexieren", variant="primary", size="lg"
                    )

                with gr.Row():
                    upload_output = gr.Textbox(
                        label="Upload-Status",
                        lines=10,
                        max_lines=15,
                        show_label=True,
                        interactive=False,
                    )

                # Event handler für Upload
                upload_btn.click(
                    fn=upload_document,
                    inputs=[file_upload],
                    outputs=[upload_output],
                    show_progress=True,
                )

                # Info-Box
                gr.Markdown("""
                **📋 Unterstützte Dateiformate:**
                - PDF-Dokumente (.pdf)
                - Text-Dateien (.txt, .md)
                - Word-Dokumente (.docx, .doc)
                - HTML-Dateien (.html)
                
                **ℹ️ Hinweise:**
                - Dokumente werden automatisch in semantische Abschnitte unterteilt
                - Die Indexierung kann je nach Dateigröße einige Sekunden dauern
                - Indexierte Dokumente stehen sofort für Anfragen zur Verfügung
                """)

        # Footer
        gr.Markdown("---")
        gr.Markdown("*Powered by Adaptive Response Engine | Web Search + RAG + KI*")

    return interface


def main():
    """Hauptfunktion für das Gradio-Interface"""

    # System in separatem Thread initialisieren
    def init_system():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(initialize_system())
            logger.info(result)
        finally:
            loop.close()

    # Initialisierung starten
    import threading

    init_thread = threading.Thread(target=init_system)
    init_thread.daemon = True
    init_thread.start()

    # Interface erstellen und starten
    interface = create_interface()

    # Server starten
    interface.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        max_threads=10,
    )


if __name__ == "__main__":
    main()
