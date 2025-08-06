from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import sys
import logging

# Configure logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

from models import HackRXRequest, HackRXResponse, HealthResponse
from advanced_rag_pipeline import AdvancedRAGPipeline
from config import get_settings

settings = get_settings()

# Global Advanced RAG pipeline instance
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_pipeline
    
    # Startup
    logger.info("Initializing Advanced RAG Pipeline...")
    try:
        rag_pipeline = AdvancedRAGPipeline()
        logger.info("Advanced RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Advanced RAG Pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Advanced RAG Pipeline...")

# Create FastAPI app
app = FastAPI(
    title="Advanced PDF QA Pipeline",
    description="Production-ready RAG-based PDF Question Answering system using Qwen3-4B and Qdrant",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced PDF QA Pipeline",
        "version": "2.0.0",
        "description": "RAG-based PDF Question Answering using Qwen3-4B and Qdrant",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
        
        # Get health status from pipeline
        health_status = await rag_pipeline.health_check()
        
        return HealthResponse(
            status=health_status["rag_pipeline"],
            services=health_status.get("services", {}),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Main endpoint: Process PDF document and answer questions using RAG pipeline
    """
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
        
        # Validate request
        if len(request.questions) > settings.max_questions:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many questions. Maximum allowed: {settings.max_questions}"
            )
        
        logger.info(f"Processing request for document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        # Process document and generate answers using RAG pipeline with timeout
        try:
            answers = await asyncio.wait_for(
                rag_pipeline.process_document_and_answer(
                    document_url=str(request.documents),
                    questions=request.questions
                ),
                timeout=settings.api_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {settings.api_timeout} seconds")
            raise HTTPException(
                status_code=408, 
                detail=f"Request timed out. Processing took longer than {settings.api_timeout} seconds."
            )
        
        logger.info("Request processed successfully")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document and questions")

@app.get("/api/v1/document/info")
async def get_document_info(
    document_url: str,
    token: str = Depends(verify_token)
):
    """Get information about a processed document"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
        
        info = await rag_pipeline.get_document_info(document_url)
        return info
        
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document information")

@app.delete("/api/v1/document/clear")
async def clear_document(
    document_url: str,
    token: str = Depends(verify_token)
):
    """Clear a document from the system"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
        
        result = await rag_pipeline.clear_document(document_url)
        return result
        
    except Exception as e:
        logger.error(f"Failed to clear document: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear document")

@app.get("/api/v1/stats")
async def get_system_stats(token: str = Depends(verify_token)):
    """Get system statistics"""
    try:
        if rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
        
        # Get vector store collection info
        collection_info = await rag_pipeline.vector_store.get_collection_info()
        
        return {
            "vector_store": collection_info,
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k_chunks": settings.top_k_chunks,
                "similarity_threshold": settings.similarity_threshold,
                "context_window": settings.context_window
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Advanced PDF QA Pipeline...")
    logger.info(f"Server will run on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
