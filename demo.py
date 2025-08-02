"""
Demo-Script für das Adaptive Response Engine System
Zeigt die Grundfunktionalität des Agent Systems
"""

import asyncio
import os
import logging
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_adaptive_response_engine():
    """Demonstriert die Funktionalität des Adaptive Response Engine Systems."""

    print("=" * 80)
    print("ADAPTIVE RESPONSE ENGINE - SYSTEM DEMO")
    print("=" * 80)

    try:
        # Importiere das Agent System
        from agents.adaptive_response_engine import AdaptiveResponseEngine

        print("✓ Adaptive Response Engine erfolgreich importiert")

        # Mock MCP Tools für Demo
        mock_mcp_tools = {
            "rag_service": None,  # Wird für Demo als None gesetzt
            "duck_searcher": None,
            "headless_browser": None,
            "ntp_time": None,
        }

        # Erstelle Engine-Instanz
        engine = AdaptiveResponseEngine(
            mcp_tools=mock_mcp_tools,
            max_iterations=3,
            quality_threshold=85.0,  # Niedrigere Schwelle für Demo
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        print("✓ Engine-Instanz erstellt")

        # Initialisiere das System
        print("\n🔄 Initialisiere Agent System...")
        await engine.initialize()
        print("✓ Agent System erfolgreich initialisiert")

        # Zeige System Status
        print("\n📊 System Status:")
        status = await engine.get_system_status()
        print(f"   - Initialisiert: {status['initialized']}")
        print(f"   - Läuft: {status['running']}")
        print(
            f"   - Agents verfügbar: {len([k for k, v in status['agents'].items() if v])}"
        )
        print(f"   - Max Iterationen: {status['configuration']['max_iterations']}")
        print(
            f"   - Qualitätsschwelle: {status['configuration']['quality_threshold']}%"
        )

        # Demo Queries
        demo_queries = [
            "Was ist maschinelles Lernen?",
            "Erkläre mir die Grundlagen der Quantenphysik.",
            "Wie funktioniert eine Blockchain?",
        ]

        print("\n" + "=" * 80)
        print("QUERY VERARBEITUNG DEMO")
        print("=" * 80)

        for i, query in enumerate(demo_queries, 1):
            print(f"\n🔍 Demo Query {i}: {query}")
            print("-" * 60)

            # Verarbeite Query ohne A2A (direkter Modus für Demo)
            result = await engine.process_query(
                query=query, context={"demo_mode": True}, use_a2a=False
            )

            # Zeige Ergebnisse
            print(f"✅ Verarbeitung abgeschlossen!")
            print(f"   📝 Antwort: {result['response'][:200]}...")
            print(f"   ⭐ Qualität: {result['quality_score']:.1f}%")
            print(f"   🔄 Iterationen: {result['iterations']}")
            print(f"   ⏱️  Zeit: {result['total_processing_time']:.2f}s")
            print(
                f"   ✅ Erfolgreich: {result.get('metadata', {}).get('success', False)}"
            )

            if i < len(demo_queries):
                print("\n⏳ Kurze Pause vor nächster Query...")
                await asyncio.sleep(2)

        # Performance Report
        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)

        performance = await engine.get_performance_report()
        if "query_processing" in performance:
            qp = performance["query_processing"]
            print(f"📈 Gesamt Queries: {qp['total_queries']}")
            print(f"⏱️  Ø Verarbeitungszeit: {qp['average_processing_time']:.2f}s")
            print(f"⭐ Ø Qualität: {qp['average_quality_score']:.1f}%")
            print(f"🔄 Ø Iterationen: {qp['average_iterations']:.1f}")
            print(f"✅ Erfolgsrate: {qp['success_rate']:.1f}%")

        # System herunterfahren
        print("\n🔄 Fahre System herunter...")
        await engine.shutdown()
        print("✅ System erfolgreich heruntergefahren")

        print("\n" + "=" * 80)
        print("DEMO ERFOLGREICH ABGESCHLOSSEN!")
        print("=" * 80)

    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        print("   Stelle sicher, dass alle Abhängigkeiten installiert sind.")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        logger.exception("Demo-Fehler")


async def demo_individual_agents():
    """Demonstriert die einzelnen Agent-Komponenten."""

    print("\n" + "=" * 80)
    print("EINZELNE AGENTS DEMO")
    print("=" * 80)

    try:
        # Teste Query Analysis Agent
        print("\n🔍 Query Analysis Agent Demo:")
        from agents.query_analysis_agent import QueryAnalysisAgent

        query_agent = QueryAnalysisAgent()

        test_query = "Wie funktioniert maschinelles Lernen?"
        analysis = await query_agent.analyze_query(test_query)

        print(f"   Query: {test_query}")
        print(f"   Intent: {analysis.get('intent', 'N/A')}")
        print(f"   Komplexität: {analysis.get('complexity_score', 'N/A')}/10")
        print(
            f"   Benötigte Quellen: {', '.join(analysis.get('required_sources', []))}"
        )

        # Teste Quality Review Agent
        print("\n⭐ Quality Review Agent Demo:")
        from agents.quality_review_agent import QualityReviewAgent

        quality_agent = QualityReviewAgent()

        sample_response = "Maschinelles Lernen ist ein Teilbereich der KI, bei dem Computer aus Daten lernen, ohne explizit programmiert zu werden."

        evaluation = await quality_agent.evaluate_response(
            test_query, sample_response, analysis, {}
        )

        print(f"   Antwort: {sample_response}")
        print(f"   Gesamt-Score: {evaluation.get('overall_score', 'N/A')}%")
        print(f"   Vollständigkeit: {evaluation.get('completeness_score', 'N/A')}%")
        print(f"   Genauigkeit: {evaluation.get('accuracy_score', 'N/A')}%")
        print(f"   Akzeptabel: {evaluation.get('is_acceptable', False)}")

        print("\n✅ Einzelne Agents erfolgreich getestet!")

    except Exception as e:
        print(f"❌ Fehler bei Agent-Tests: {e}")
        logger.exception("Agent-Test-Fehler")


def main():
    """Hauptfunktion für Demo-Ausführung."""

    # Prüfe Umgebung
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ WARNUNG: OPENAI_API_KEY nicht gesetzt!")
        print("   Das System wird mit Mock-Responses arbeiten.")
        print("   Setze die Umgebungsvariable für vollständige Funktionalität.\n")

    print("🚀 Starte Adaptive Response Engine Demo...")

    try:
        # Führe beide Demos aus
        asyncio.run(demo_individual_agents())
        asyncio.run(demo_adaptive_response_engine())

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"\n❌ Demo-Fehler: {e}")
        logger.exception("Hauptdemo-Fehler")


if __name__ == "__main__":
    main()
