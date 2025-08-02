import os
import uvicorn
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi_mcp import FastApiMCP
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Import Service Classes ---
from mcp_services.mcp_website.headless_browser import HeadlessBrowserExtractor
from mcp_services.mcp_time.ntp_time import NtpTime
from mcp_services.mcp_search.duck_search import (
    DuckDuckGoSearcher,
    DuckDuckGoSearchResults,
)
from mcp_services.mcp_qdrant.qdrant_serve import IntegratedKnowledgeSystem

# --- Agent System Imports ---
from agents.adaptive_response_engine import AdaptiveResponseEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_server.main")

# --- Load Environment Variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(
        f".env file not found at {dotenv_path}. Relying on system environment variables."
    )

# --- Instantiate Services ---
headless_browser: Optional[HeadlessBrowserExtractor] = None
ntp_time: Optional[NtpTime] = None
duck_searcher: Optional[DuckDuckGoSearcher] = None
rag_service: Optional[IntegratedKnowledgeSystem] = None
adaptive_response_engine: Optional[AdaptiveResponseEngine] = None

try:
    logger.info("Initializing HeadlessBrowserExtractor...")
    headless_browser = HeadlessBrowserExtractor()

    logger.info("Initializing NtpTime...")
    ntp_time = NtpTime()

    logger.info("Initializing DuckDuckGoSearcher...")
    duck_searcher = DuckDuckGoSearcher()

    logger.info("Initializing RAG Service...")
    rag_collection_name = os.getenv("RAG_COLLECTION_NAME", "integrated_knowledge")
    rag_service = IntegratedKnowledgeSystem(collection_name=rag_collection_name)

    logger.info("Initializing Adaptive Response Engine...")

    # Sammle verfügbare MCP-Tools
    mcp_tools = {
        "headless_browser": headless_browser,
        "ntp_time": ntp_time,
        "duck_searcher": duck_searcher,
        "rag_service": rag_service,
    }

    # Initialisiere Adaptive Response Engine
    adaptive_response_engine = AdaptiveResponseEngine(
        mcp_tools=mcp_tools,
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "95.0")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

except ImportError as e:
    logger.error(f"Import error: {e}. Dependencies might be missing.")
    raise SystemExit(f"Failed to import service component: {e}") from e
except Exception as e:
    logger.exception(f"Unexpected error during service initialization: {e}")
    raise SystemExit(f"Unexpected error during service initialization: {e}") from e


# --- Pydantic Models ---


class ExtractTextRequest(BaseModel):
    url: str = Field(..., description="URL of the website to extract text from.")


class ExtractTextResponse(BaseModel):
    url: str
    text_content: Optional[str] = None
    error: Optional[str] = None


class NtpTimeResponse(BaseModel):
    current_time_utc: Optional[str] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query.")
    max_results: int = Field(
        5, description="Maximum number of search results to return.", gt=0, le=20
    )


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query for knowledge base search.")
    metadata_filter: Optional[dict] = None
    limit: int = Field(10, description="Maximum number of results to return.")


class QueryResponse(BaseModel):
    results: dict
    error: Optional[str] = None


class IntakeRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to index.")


class IntakeResponse(BaseModel):
    status: str
    error: Optional[str] = None


class AgentProcessRequest(BaseModel):
    query: str = Field(..., description="User query to process.")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context.")
    use_a2a: bool = Field(True, description="Use A2A coordination mode.")


class AgentProcessResponse(BaseModel):
    response: str
    quality_score: float
    iterations: int
    total_processing_time: float
    success: bool
    error: Optional[str] = None


# --- FastAPI Setup ---
app = FastAPI(
    title="Adaptive Response Engine MCP Server",
    description="Advanced Agent System with MCP integration, A2A protocol, and iterative quality improvement.",
    version="1.0.0",
)


# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Initialisiert das Agent System beim Server-Start."""
    if adaptive_response_engine:
        try:
            await adaptive_response_engine.initialize()
            logger.info(
                "Adaptive Response Engine erfolgreich beim Server-Start initialisiert"
            )
        except Exception as e:
            logger.error(f"Fehler bei Agent System Initialisierung: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Fährt das Agent System beim Server-Stopp herunter."""
    if adaptive_response_engine:
        try:
            await adaptive_response_engine.shutdown()
            logger.info("Adaptive Response Engine erfolgreich heruntergefahren")
        except Exception as e:
            logger.error(f"Fehler beim Agent System Shutdown: {e}")


# --- Standard Routes ---
@app.get("/", include_in_schema=False)
async def root():
    return PlainTextResponse("Adaptive Response Engine MCP Server is running.")


@app.get("/health", include_in_schema=False)
async def health_check():
    return JSONResponse({"status": "ok"})


@app.get("/system-status")
async def get_system_status():
    """Gibt den aktuellen Systemstatus zurück."""
    if adaptive_response_engine:
        status = await adaptive_response_engine.get_system_status()
        return JSONResponse(status)
    else:
        return JSONResponse({"error": "Agent System nicht verfügbar"}, status_code=503)


# --- Helper Functions ---
def check_service(service_instance, service_name: str):
    if service_instance is None:
        logger.error(f"Attempted to use unavailable {service_name} service.")
        raise HTTPException(
            status_code=503,
            detail=f"{service_name} service is not configured or failed to initialize.",
        )


def check_rag_service():
    check_service(rag_service, "RAG")


# --- Service Endpoints ---


@app.post(
    "/extract-text",
    response_model=ExtractTextResponse,
    summary="Extract Text Content from URL",
    operation_id="extract_website_text",
    include_in_schema=False,
)
async def extract_text_endpoint(request_data: ExtractTextRequest):
    """Extracts the main textual content from a given website URL."""
    check_service(headless_browser, "Headless Browser")
    logger.info(f"API: Received request to extract text from URL: {request_data.url}")
    try:
        text = await headless_browser.extract_text(request_data.url)
        return ExtractTextResponse(url=request_data.url, text_content=text)
    except Exception as e:
        logger.exception(f"Error extracting text from {request_data.url}: {e}")
        return ExtractTextResponse(url=request_data.url, error=str(e))


@app.get(
    "/current-time",
    response_model=NtpTimeResponse,
    summary="Get Current UTC Time from NTP",
    operation_id="get_current_time_utc",
    include_in_schema=False,
)
async def current_time_endpoint():
    """Gets the current accurate UTC time from an NTP server."""
    check_service(ntp_time, "NTP Time")
    logger.info("API: Received request for current time.")
    try:
        current_time = await ntp_time.get_current_time_iso()
        return NtpTimeResponse(current_time_utc=current_time)
    except Exception as e:
        logger.exception(f"Error getting NTP time: {e}")
        return NtpTimeResponse(error=str(e))


@app.post(
    "/search",
    response_model=DuckDuckGoSearchResults,
    summary="Perform DuckDuckGo Search",
    operation_id="duckduckgo_search",
    include_in_schema=False,
)
async def search_endpoint(request_data: SearchRequest):
    """Performs a web search using DuckDuckGo."""
    check_service(duck_searcher, "DuckDuckGo Search")
    logger.info(f"API: Received search request for query: '{request_data.query}'")
    try:
        results = duck_searcher.search(
            request_data.query, max_results=request_data.max_results
        )
        return results
    except Exception as e:
        logger.exception(f"Error performing search for '{request_data.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post(
    "/query-knowledge",
    response_model=QueryResponse,
    summary="Query Knowledge Base",
    operation_id="query_knowledge",
    include_in_schema=False,
)
async def query_knowledge_endpoint(request_data: QueryRequest):
    """Queries the knowledge base using RAG."""
    check_rag_service()
    try:
        results = rag_service.query_knowledge(
            query=request_data.query,
            limit=request_data.limit,
            metadata_filter=request_data.metadata_filter,
        )
        return QueryResponse(results={"results": results})
    except Exception as e:
        logger.exception(f"Error querying knowledge base: {e}")
        return QueryResponse(results={}, error=str(e))


@app.post(
    "/index-document",
    response_model=IntakeResponse,
    summary="Index Document in Knowledge Base",
    operation_id="index_document",
    include_in_schema=False,
)
async def index_document_endpoint(request_data: IntakeRequest):
    """Indexes a document in the knowledge base."""
    check_rag_service()
    try:
        rag_service.index_document(request_data.file_path)
        return IntakeResponse(status="success")
    except Exception as e:
        logger.exception(f"Error indexing document: {e}")
        return IntakeResponse(status="failed", error=str(e))


# --- Main Agent System Endpoint ---
@app.post(
    "/process-query",
    response_model=AgentProcessResponse,
    summary="Process User Query with Adaptive Response Engine",
    operation_id="process_user_query",
    include_in_schema=False,
)
async def process_query_endpoint(request_data: AgentProcessRequest):
    """Processes a user query through the adaptive response agent system."""
    check_service(adaptive_response_engine, "Adaptive Response Engine")
    logger.info(
        f"API: Received query processing request: '{request_data.query[:50]}...'"
    )

    try:
        result = await adaptive_response_engine.process_query(
            query=request_data.query,
            context=request_data.context,
            use_a2a=request_data.use_a2a,
        )

        return AgentProcessResponse(
            response=result["response"],
            quality_score=result["quality_score"],
            iterations=result["iterations"],
            total_processing_time=result["total_processing_time"],
            success=result.get("metadata", {}).get("success", False),
            error=result.get("error"),
        )
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        return AgentProcessResponse(
            response="Entschuldigung, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten.",
            quality_score=0.0,
            iterations=0,
            total_processing_time=0.0,
            success=False,
            error=str(e),
        )


# --- FastAPI MCP Integration ---
server_host = os.environ.get("SERVER_HOST", "localhost")
server_port = os.environ.get("SERVER_PORT", "8000")
try:
    port_num = int(server_port)
except ValueError:
    port_num = 8000
    logger.warning(f"Invalid SERVER_PORT '{server_port}', using default {port_num}.")

server_scheme = os.environ.get("SERVER_SCHEME", "http")
base_url = f"{server_scheme}://{server_host}:{port_num}"
logger.info(f"Configuring MCP with base_url: {base_url}")

mcp = FastApiMCP(
    app,
    name="Adaptive Response Engine MCP",
    description="Advanced Agent System with MCP integration, A2A protocol, iterative quality improvement, RAG with Qdrant, DuckDuckGo search, web content extraction, and time services.",
)
mcp.mount()


# --- Run Server ---
if __name__ == "__main__":
    logger.info(
        f"Starting Adaptive Response Engine MCP Server on host 0.0.0.0:{port_num}"
    )

    reload_enabled = os.getenv("UVICORN_RELOAD", "true").lower() == "true"
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info").lower()

    uvicorn.run(
        "mcp_main:app",
        host="0.0.0.0",
        port=port_num,
        reload=reload_enabled,
        log_level=log_level,
    )
