# Adaptive Response Engine

Ein fortschrittliches Multi-Agent-System für hochqualitative, iterative Antwortgenerierung mit Web-Interface. Das System kombiniert lokale Wissensbasis (RAG), Web-Suche und adaptive Qualitätskontrolle für umfassende und präzise Antworten.

## Hauptfeatures

### Multi-Agent-Architektur
- **QueryAnalysisAgent**: Analysiert Nutzereingaben auf Intent, Komplexität und benötigte Informationsquellen
- **ResponseGenerationAgent**: Generiert Antworten durch intelligente Synthese aus mehreren Informationsquellen  
- **QualityReviewAgent**: Bewertet Antwortqualität in 4 Dimensionen (Vollständigkeit, Genauigkeit, Relevanz, Kohärenz)
- **IterationController**: Steuert iterative Verbesserungsprozesse bis 95% Qualitätsschwelle erreicht ist
- **A2ACoordinator**: Implementiert Agent-to-Agent Protokoll für koordinierte Multi-Agent-Operationen

### Informationsquellen
- **RAG-System**: RAG mit Qdrant als Vektordatenbank für lokale Wissensbasis-Abfragen
- **Web-Suche**: DuckDuckGo-Integration mit intelligenter Fallback-Strategie für aktuelle Informationen
- **Web-Extraktion**: Headless Browser (Playwright) für Website-Inhalte
- **Zeit-Service**: NTP-basierte präzise Zeitangaben für zeitkritische Anfragen

### Gradio Web-Interface
- **Chat-Interface**: Benutzerfreundliche Web-Oberfläche für Anfragen
- **Dokument-Upload**: Einfache Erweiterung der Wissensbasis über PDF, TXT, DOCX, etc.
- **Monochrome Design**: Professionelles, ablenkungsfreies Interface
- **Echtzeit-Verarbeitung**: Live-Updates während der Antwortgenerierung

### Model Context Protocol (MCP) Integration
- Vollständige MCP-Unterstützung für externe Tool-Integration
- FastAPI-basierter MCP Server mit automatischer Tool-Discovery
- Nahtlose Integration in bestehende MCP-Ökosysteme

## Systemarchitektur

```
Adaptive Response Engine
├── Web Interface (Gradio)
│   ├── Chat-Tab (Anfrageverarbeitung)
│   └── Upload-Tab (Dokumentenindexierung)
│
├── Agent System
│   ├── Query Analysis Agent (Intent & Komplexität)
│   ├── Response Generation Agent (Multi-Source Synthese)
│   ├── Quality Review Agent (4D-Qualitätsbewertung)
│   ├── Iteration Controller (Verbesserungsschleife)
│   └── A2A Coordinator (Agent-Koordination)
│
└── MCP Services
    ├── Qdrant RAG Service (Wissensbasis mit Embeddings)
    ├── DuckDuckGo Search (Multi-Backend Web-Suche)
    ├── Headless Browser (Website-Textextraktion)
    └── NTP Time Service (Präzise Zeitdienste)
```

## Installation

### Voraussetzungen
- Python 3.10+
- UV Package Manager
- Qdrant Vector Database (Docker empfohlen)
- Ollama (für lokale LLM- und Embedding-Models)

### Setup
```bash
# Repository klonen
git clone <repository-url>
cd Adaptive_Response_Engine

# Dependencies installieren
uv sync

# Qdrant starten (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Ollama installieren und Models laden
ollama pull qwen2.5:latest
ollama pull bge-m3:latest

# Umgebungsvariablen konfigurieren (optional)
cp .env.example .env
# Bearbeite .env für spezielle Konfigurationen
```

## Verwendung

### Web-Interface starten
```bash
# Hauptanwendung starten
uv run python gradio_frontend.py
# oder alternativ
uv run python start_frontend.py

# Interface öffnen: http://localhost:7860
```

### Chat-Funktionen
1. **Fragen stellen**: Eingabe in das Chat-Interface
2. **Dokumentenupload**: PDF/TXT-Dateien über Upload-Tab hinzufügen
3. **Quellenverweise**: Automatische Anzeige relevanter URLs und Quellen

### MCP Server (optional)
```bash
# MCP Server für externe Integration
uv run mcp_main.py

# Demo-Script für Entwicklung
uv run demo.py
```

### API Endpunkte

#### Query-Verarbeitung
```bash
POST /process-query
{
    "query": "Erkläre mir maschinelles Lernen",
    "context": {"user_level": "beginner"},
    "use_a2a": true
}
```

#### System-Status
```bash
GET /system-status
```

## Konfiguration

### Umgebungsvariablen
```bash
# LLM-Konfiguration (Standard: Ollama)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:latest
EMBEDDING_MODEL=bge-m3:latest

# Agent System
MAX_ITERATIONS=3                # Maximale Verbesserungsiterationen
QUALITY_THRESHOLD=95.0          # Mindestqualität in Prozent

# Qdrant Konfiguration
QDRANT_URL=http://localhost:6333
RAG_COLLECTION_NAME=adaptive_response_MANHATTAN

# Server Konfiguration
SERVER_HOST=0.0.0.0
SERVER_PORT=7860
```

### System-Parameter
- **Qualitätsschwelle**: 95% (konfigurierbar)
- **Maximale Iterationen**: 3 pro Anfrage
- **Embedding-Model**: bge-m3:latest (mehrsprachig)
- **LLM-Model**: qwen2.5:latest (Antwortgenerierung)
- **Web-Suche**: Automatisches Fallback zwischen DuckDuckGo, Google, Brave, Bing, Yandex

## MCP Tools

Verfügbare Tools über Model Context Protocol:
- `process_user_query`: Hauptverarbeitung über Agent-System
- `extract_website_text`: Website-Inhalte extrahieren
- `duckduckgo_search`: Intelligente Web-Suche
- `query_knowledge`: Lokale Wissensbasis abfragen
- `index_document`: Dokumente zur Wissensbasis hinzufügen
- `get_current_time_utc`: Aktuelle Zeitangaben

## Entwicklung

### Projektstruktur
```
├── gradio_frontend.py         # Hauptanwendung (Web-Interface)
├── start_frontend.py          # Einfacher Launcher
├── agents/                    # Agent System
│   ├── adaptive_response_engine.py    # Hauptorchestrator
│   ├── query_analysis_agent.py        # Query-Analyse
│   ├── response_generation_agent.py   # Antwort-Generierung  
│   ├── quality_review_agent.py        # Qualitätsbewertung
│   ├── iteration_controller.py        # Iterationssteuerung
│   └── a2a_coordinator.py             # Agent-Koordination
├── mcp_services/              # MCP Service Implementierungen
│   ├── mcp_qdrant/           # RAG mit Qdrant & Embeddings
│   ├── mcp_search/           # Multi-Backend Web-Suche
│   ├── mcp_time/             # NTP Zeit-Service
│   └── mcp_website/          # Playwright Web-Extraktion
├── mcp_main.py               # MCP-Server für externe Integration
└── requirements.txt          # Python Dependencies
```

### Systemerweiterungen
1. **Neue Agents**: Implementiere AgentRole Interface und registriere im A2ACoordinator
2. **Neue MCP Services**: Service in mcp_services/ hinzufügen und in gradio_frontend.py registrieren
3. **Neue Qualitätsdimensionen**: QualityReviewAgent.evaluate_response() erweitern
4. **UI-Komponenten**: Gradio-Interface in create_interface() anpassen

### Debugging & Monitoring
```bash
# Ausführliche Logs aktivieren
export LOGGING_LEVEL=DEBUG

# System-Status prüfen
curl http://localhost:7860/system-status

# Performance-Report
# Verfügbar über AdaptiveResponseEngine.get_performance_report()
```

## Performance & Monitoring

Das System bietet umfangreiches Performance-Monitoring:
- **Query-Verarbeitungszeiten**: Durchschnittliche Antwortzeiten pro Agent
- **Qualitäts-Trends**: Entwicklung der Antwortqualität über Zeit
- **Iterations-Statistiken**: Erfolgsraten und Verbesserungszyklen
- **Quellen-Analytics**: Nutzung von RAG vs. Web-Suche
- **Agent-Koordination**: Effizienz der Multi-Agent-Zusammenarbeit

Monitoring über:
- Terminal-Logs (strukturiertes Logging)
- `/system-status` API-Endpunkt
- Performance-Reports über Python-API

## Troubleshooting

### Häufige Probleme
1. **RAG-System nicht verfügbar**: Qdrant-Server prüfen, Embedding-Model verificieren
2. **Web-Suche fehlgeschlagen**: Internet-Verbindung und DuckDuckGo-Verfügbarkeit prüfen
3. **LLM-Anfragen scheitern**: Ollama-Server Status und Model-Verfügbarkeit prüfen
4. **Gradio startet nicht**: Port 7860 Verfügbarkeit und Python-Environment prüfen

### Logs und Debugging
```bash
# Detaillierte Logs
tail -f logs/adaptive_response_engine.log

# Ollama-Status prüfen
ollama list
ollama ps

# Qdrant-Status prüfen
curl http://localhost:6333/health
```

## Lizenz

Dieses Projekt steht unter der AGPLv3 Lizenz - siehe LICENSE-Datei für Details.

