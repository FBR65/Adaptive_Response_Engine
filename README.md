# Adaptive Response Engine

Ein fortschrittliches Agent-System, das Nutzereingaben analysiert und verbessert. Es prüft Antworten iterativ auf Genauigkeit und erzwingt ggf. neue Antworten, bis die Anforderungen des Nutzers zu mindestens 95% erfüllt sind.

## 🚀 Hauptfeatures

### Agent System
- **QueryAnalysisAgent**: Analysiert Nutzereingaben auf Intent, Komplexität und benötigte Informationsquellen
- **ResponseGenerationAgent**: Generiert Antworten durch Synthese aus mehreren Informationsquellen  
- **QualityReviewAgent**: Bewertet Antwortqualität in 4 Dimensionen (Vollständigkeit, Genauigkeit, Relevanz, Kohärenz)
- **IterationController**: Steuert den iterativen Verbesserungsprozess bis 95% Qualitätsschwelle erreicht ist
- **A2ACoordinator**: Implementiert Agent-to-Agent Protokoll für koordinierte Multi-Agent-Operationen

### Informationsquellen
- **RAG-System**: Colpali-RAG mit Qdrant als Vektordatenbank für Wissensbasis-Abfragen
- **Web-Suche**: DuckDuckGo-Integration mit Selenium für aktuelle Informationen
- **Web-Extraktion**: Headless Browser für Website-Inhalte
- **Zeit-Service**: NTP-basierte präzise Zeitangaben

### MCP Integration
- Vollständige Model Context Protocol (MCP) Unterstützung
- FastAPI-basierter MCP Server mit Tool-Discovery
- Automatische Endpunkt-Erkennung und -Integration

## 🏗️ Systemarchitektur

```
Adaptive Response Engine
├── Query Analysis Agent (Intent & Komplexität)
├── Response Generation Agent (Multi-Source Synthese)
├── Quality Review Agent (4D-Qualitätsbewertung)
├── Iteration Controller (Verbesserungsschleife)
└── A2A Coordinator (Agent-Koordination)

MCP Tools Integration:
├── Qdrant RAG Service (Wissensbasis)
├── DuckDuckGo Search (Web-Suche)
├── Headless Browser (Web-Extraktion)
└── NTP Time Service (Zeitdienst)
```

## 📦 Installation

```bash
# Dependencies installieren
uv sync

# Umgebungsvariablen konfigurieren
cp .env.example .env
# Bearbeite .env mit deinen API-Keys
```

## 🚀 Verwendung

### Server starten
```bash
# MCP Server starten
uv run mcp_main.py

# Demo ausführen
uv run demo.py
```

### API Endpunkte

#### Hauptendpunkt - Query Verarbeitung
```bash
POST /process-query
{
    "query": "Erkläre mir maschinelles Lernen",
    "context": {"user_level": "beginner"},
    "use_a2a": true
}
```

#### System Status
```bash
GET /system-status
```

### MCP Tools
Die folgenden Tools sind über MCP verfügbar:
- `process_user_query`: Hauptendpunkt für Agent-System
- `extract_website_text`: Web-Inhalte extrahieren
- `duckduckgo_search`: Web-Suche durchführen
- `query_knowledge`: Wissensbasis abfragen
- `index_document`: Dokument in Wissensbasis hinzufügen
- `get_current_time_utc`: Aktuelle Zeit abrufen

## ⚙️ Konfiguration

### Umgebungsvariablen
```bash
# OpenAI API für LLM-Services
OPENAI_API_KEY=sk-...

# Agent System Konfiguration  
MAX_ITERATIONS=3                # Max. Iterationen pro Query
QUALITY_THRESHOLD=95.0          # Mindestqualität in %

# Qdrant Konfiguration
QDRANT_URL=http://localhost:6333
RAG_COLLECTION_NAME=integrated_knowledge

# Server Konfiguration
SERVER_HOST=localhost
SERVER_PORT=8000
SERVER_SCHEME=http
```

### Agent System Parameter
- **Qualitätsschwelle**: 95% (konfigurierbar)
- **Max. Iterationen**: 3 (konfigurierbar)  
- **A2A-Modus**: Aktivierbar für koordinierte Agent-Operationen
- **Fallback-Modi**: Bei Fehlern automatischer Wechsel zu direkter Verarbeitung

## 🔧 Entwicklung

### Projektstruktur
```
├── agents/                    # Agent System
│   ├── adaptive_response_engine.py    # Hauptorchestrator
│   ├── query_analysis_agent.py        # Query-Analyse
│   ├── response_generation_agent.py   # Antwort-Generierung  
│   ├── quality_review_agent.py        # Qualitätsbewertung
│   ├── iteration_controller.py        # Iterationssteuerung
│   └── a2a_coordinator.py             # Agent-Koordination
├── mcp_services/              # MCP Service Implementierungen
│   ├── mcp_qdrant/           # RAG mit Qdrant
│   ├── mcp_search/           # DuckDuckGo Suche
│   ├── mcp_time/             # NTP Zeit-Service
│   └── mcp_website/          # Web-Extraktion
├── mcp_main.py               # Haupt-MCP-Server
├── demo.py                   # Demo-Script
└── README.md                 # Diese Datei
```

### Erweitern des Systems
1. **Neue Agents**: Implementiere AgentRole und registriere im A2ACoordinator
2. **Neue MCP Tools**: Füge Service in mcp_services/ hinzu und registriere in mcp_main.py
3. **Neue Qualitätsdimensionen**: Erweitere QualityReviewAgent.evaluate_response()

## 📊 Performance & Monitoring

Das System bietet umfangreiches Performance-Monitoring:
- Query-Verarbeitungszeiten
- Qualitäts-Scores über Zeit
- Iterations-Statistiken  
- Agent-Koordinations-Metriken
- Erfolgsraten und Trends

Zugriff über `/system-status` Endpunkt oder AdaptiveResponseEngine.get_performance_report().

## 🤝 Contributing

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/amazing-feature`)
3. Committe deine Änderungen (`git commit -m 'Add amazing feature'`)
4. Push zum Branch (`git push origin feature/amazing-feature`)
5. Öffne eine Pull Request

## 📝 Lizenz

Dieses Projekt steht unter der MIT Lizenz - siehe LICENSE Datei für Details.
