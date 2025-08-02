"""
Startskript für das Adaptive Response Engine Gradio Frontend
"""

import subprocess
import sys
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Startet das Gradio Frontend"""
    try:
        logger.info("🚀 Starte Adaptive Response Engine Frontend...")
        logger.info("📱 Das Interface wird unter http://localhost:7860 verfügbar sein")
        logger.info("⏳ Systeminitialisierung läuft im Hintergrund...")

        # Starte das Gradio Frontend
        subprocess.run([sys.executable, "gradio_frontend.py"], check=True)

    except KeyboardInterrupt:
        logger.info("👋 Frontend beendet")
    except Exception as e:
        logger.error(f"❌ Fehler beim Starten des Frontends: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
