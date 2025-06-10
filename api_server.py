#!/usr/bin/env python3
"""
smurf API Server
FastAPI-based REST API for the smurf RAG system
"""

import os
import asyncio
import logging
import sys
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
import uvicorn
from dotenv import load_dotenv

from src.core.database import Database
from src.processors.web import WebProcessor
from src.processors.github import GitHubProcessor
from src.processors.router import ProcessorRouter


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/api.log') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global application state
app_state = {
    'db': None,
    'router': None,
    'initialized': False
}


# Pydantic models
class CrawlRequest(BaseModel):
    url: HttpUrl
    smart_crawl: bool = False
    depth: int = 1

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    source_filter: Optional[str] = None

class CrawlResponse(BaseModel):
    success: bool
    message: str
    url: str
    chunks: int = 0

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    count: int

class SourcesResponse(BaseModel):
    sources: List[Dict[str, Any]]
    count: int

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    components: Dict[str, str]


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("🚀 Starting smurf API server...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database connection...")
        app_state['db'] = Database()
        await app_state['db'].initialize()
        
        # Initialize processors
        logger.info("🔧 Setting up processors...")
        web_processor = WebProcessor(app_state['db'])
        github_processor = GitHubProcessor(app_state['db'])
        app_state['router'] = ProcessorRouter([github_processor, web_processor], web_processor)
        
        app_state['initialized'] = True
        logger.info("✅ smurf API server ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize smurf API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🧹 Shutting down smurf API server...")
    
    if app_state['router']:
        for processor in app_state['router'].processors:
            if hasattr(processor, 'cleanup'):
                try:
                    await processor.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up processor: {e}")
    
    logger.info("👋 smurf API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="smurf API",
    description="API for the smurf RAG system - crawl, index, and search content",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to check initialization
def check_initialized():
    if not app_state['initialized']:
        raise HTTPException(status_code=503, detail="smurf is not initialized")
    return app_state


# API Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information."""
    components = {}
    
    if app_state['db']:
        try:
            # Test database connection
            conn = app_state['db'].get_connection()
            conn.close()
            components['database'] = 'healthy'
        except Exception:
            components['database'] = 'unhealthy'
    else:
        components['database'] = 'not_initialized'
    
    components['router'] = 'healthy' if app_state['router'] else 'not_initialized'
    
    return HealthResponse(
        status="healthy" if app_state['initialized'] else "initializing",
        components=components
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": str(asyncio.get_event_loop().time())}


@app.post("/crawl", response_model=CrawlResponse)
async def crawl_url(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    state: dict = Depends(check_initialized)
):
    """Crawl and process a URL."""
    try:
        url_str = str(request.url)
        logger.info(f"🌐 API: Processing URL: {url_str}")
        
        options = {
            'smart_crawl': request.smart_crawl,
            'depth': request.depth
        }
        
        result = await state['router'].route(url_str, options)
        
        return CrawlResponse(
            success=result.success,
            message=result.message,
            url=url_str,
            chunks=len(result.contents) if result.success else 0
        )
        
    except Exception as e:
        logger.error(f"❌ API crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    state: dict = Depends(check_initialized)
):
    """Search the knowledge base."""
    try:
        logger.info(f"🔍 API: Searching for: {request.query}")
        
        results = state['db'].search_documents(
            request.query,
            request.limit,
            source_filter=request.source_filter
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"❌ API search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    source: Optional[str] = Query(None, description="Filter by source"),
    state: dict = Depends(check_initialized)
):
    """Search the knowledge base via GET request."""
    try:
        results = state['db'].search_documents(q, limit, source_filter=source)
        return SearchResponse(query=q, results=results, count=len(results))
    except Exception as e:
        logger.error(f"❌ API search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", response_model=SourcesResponse)
async def get_sources(state: dict = Depends(check_initialized)):
    """Get all available sources."""
    try:
        sources = state['db'].get_all_sources()
        return SourcesResponse(sources=sources, count=len(sources))
    except Exception as e:
        logger.error(f"❌ API sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(state: dict = Depends(check_initialized)):
    """Get router and system statistics."""
    try:
        router_stats = state['router'].get_stats()
        processor_info = state['router'].list_processors()
        
        return {
            'router': router_stats,
            'processors': processor_info,
            'timestamp': str(asyncio.get_event_loop().time())
        }
    except Exception as e:
        logger.error(f"❌ API stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str, state: dict = Depends(check_initialized)):
    """Delete a source and all its documents."""
    try:
        # This would need to be implemented in the database class
        # For now, return not implemented
        raise HTTPException(status_code=501, detail="Source deletion not implemented")
    except Exception as e:
        logger.error(f"❌ API delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-crawl")
async def batch_crawl(
    urls: List[HttpUrl],
    background_tasks: BackgroundTasks,
    smart_crawl: bool = False,
    depth: int = 1,
    state: dict = Depends(check_initialized)
):
    """Crawl multiple URLs in batch."""
    try:
        url_strings = [str(url) for url in urls]
        logger.info(f"🌐 API: Batch processing {len(url_strings)} URLs")
        
        options = {'smart_crawl': smart_crawl, 'depth': depth}
        results = await state['router'].process_multiple(url_strings, options)
        
        successful = sum(1 for r in results if r.success)
        total_chunks = sum(len(r.contents) for r in results if r.success)
        
        return {
            'total_urls': len(url_strings),
            'successful': successful,
            'failed': len(url_strings) - successful,
            'total_chunks': total_chunks,
            'results': [
                {
                    'url': url_strings[i],
                    'success': r.success,
                    'chunks': len(r.contents) if r.success else 0,
                    'message': r.message
                }
                for i, r in enumerate(results)
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ API batch crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


def main():
    """Run the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"🌐 Starting smurf API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,  # Single worker for shared state
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()