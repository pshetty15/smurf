#!/usr/bin/env python3
"""
smurf HTTP MCP Server - Network accessible MCP server for multiple AI agents
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

from src.core.database import Database
from src.processors.web import WebProcessor
from src.processors.github import GitHubProcessor
from src.processors.router import ProcessorRouter

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

# Pydantic models for HTTP API
class ToolRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}

class ToolResponse(BaseModel):
    success: bool
    content: str
    error: bool = False

class ListToolsResponse(BaseModel):
    tools: List[Dict[str, Any]]

class SmurfHTTPMCPServer:
    """HTTP-based MCP Server for smurf - accessible by multiple AI agents."""
    
    def __init__(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.initialize()
            yield
            # Shutdown - cleanup if needed
            pass
        
        self.app = FastAPI(
            title="smurf MCP Server",
            description="HTTP-based MCP server for smurf semantic search and indexing",
            version="1.0.0",
            lifespan=lifespan
        )
        self.db = None
        self.router = None
        self.initialized = False
        
        # Add CORS middleware for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_routes()

    async def initialize(self):
        """Initialize smurf components."""
        if self.initialized:
            return
            
        try:
            logger.info("🚀 Initializing smurf HTTP MCP server...")
            
            # Load environment variables
            load_dotenv()
            
            # Initialize database
            self.db = Database()
            await self.db.initialize()
            
            # Initialize processors
            web_processor = WebProcessor(self.db)
            github_processor = GitHubProcessor(self.db)
            self.router = ProcessorRouter([github_processor, web_processor], web_processor)
            
            self.initialized = True
            logger.info("✅ smurf HTTP MCP server initialized!")
            
        except Exception as e:
            logger.error(f"Failed to initialize smurf: {e}")
            raise

    def setup_routes(self):
        """Setup HTTP routes for MCP protocol."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "smurf MCP server"}
        
        @self.app.get("/mcp/tools", response_model=ListToolsResponse)
        async def list_tools():
            """List available smurf tools (MCP protocol)."""
            tools = [
                {
                    "name": "smurf",
                    "description": """Crawl and index content from URLs into the smurf knowledge base.
                    
Supports various content types:
- Web pages and articles  
- GitHub repositories (full repo or specific files)
- Sitemaps (XML format)
- Plain text files

Examples:
- smurf https://docs.python.org/3/tutorial/
- smurf https://github.com/fastapi/fastapi
- smurf https://ai-sdk.dev/llms.txt

Returns: Success message with details about indexed content""",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to crawl and index"
                            }
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "smurf_search",
                    "description": """Search the smurf knowledge base using semantic search.
                    
Uses vector embeddings to find semantically similar content.
Returns ranked results with similarity scores.

Examples:
- smurf_search "machine learning tutorials"
- smurf_search "FastAPI authentication"
- smurf_search "Python async programming"

Returns: List of relevant content chunks with URLs and scores""",
                    "inputSchema": {
                        "type": "object", 
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "smurf_sources",
                    "description": "List all indexed sources in the smurf knowledge base",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]
            
            return ListToolsResponse(tools=tools)
        
        @self.app.post("/mcp/tools/call", response_model=ToolResponse)
        async def call_tool(request: ToolRequest):
            """Call a smurf tool (MCP protocol)."""
            if not self.initialized:
                await self.initialize()
                
            try:
                if request.name == "smurf":
                    result = await self.handle_smurf(request.arguments)
                elif request.name == "smurf_search":
                    result = await self.handle_search(request.arguments)
                elif request.name == "smurf_sources":
                    result = await self.handle_sources(request.arguments)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown tool: {request.name}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error calling tool {request.name}: {e}")
                return ToolResponse(
                    success=False,
                    content=f"Error calling tool: {str(e)}",
                    error=True
                )

    async def handle_smurf(self, arguments: Dict[str, Any]) -> ToolResponse:
        """Handle smurf tool call for crawling."""
        try:
            url = arguments.get("url")
            if not url:
                return ToolResponse(
                    success=False,
                    content="Error: URL is required",
                    error=True
                )
            
            # Process the URL
            result = await self.router.route(url)
            
            if result.success:
                message = f"✅ Successfully indexed content from: {url}\n"
                message += f"📊 Processed {len(result.contents)} content chunks\n"
                
                if result.message:
                    message += f"📝 {result.message}\n"
                if result.processor_name:
                    message += f"📄 Processor: {result.processor_name}\n"
                
                message += f"\nUse 'smurf_search' to query the indexed content."
                
                return ToolResponse(success=True, content=message)
            else:
                return ToolResponse(
                    success=False,
                    content=f"❌ Failed to process {url}: {result.message}",
                    error=True
                )
                
        except Exception as e:
            logger.error(f"Error in smurf handler: {e}")
            return ToolResponse(
                success=False,
                content=f"❌ Error processing URL: {str(e)}",
                error=True
            )

    async def handle_search(self, arguments: Dict[str, Any]) -> ToolResponse:
        """Handle search tool call."""
        try:
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            
            if not query:
                return ToolResponse(
                    success=False,
                    content="Error: Query is required",
                    error=True
                )
            
            # Perform search
            results = self.db.search_documents(query, limit)
            
            if not results:
                return ToolResponse(
                    success=True,
                    content=f"No results found for: {query}"
                )
            
            # Format results
            message = f"🔍 Search results for: {query}\n\n"
            
            for i, result in enumerate(results, 1):
                message += f"**Result {i}** (Score: {result.get('score', 0):.3f})\n"
                if result.get('title'):
                    message += f"📝 **{result['title']}**\n"
                if result.get('url'):
                    message += f"🔗 {result['url']}\n"
                message += f"📄 {result.get('content', '')[:300]}...\n\n"
            
            return ToolResponse(success=True, content=message)
            
        except Exception as e:
            logger.error(f"Error in search handler: {e}")
            return ToolResponse(
                success=False,
                content=f"❌ Search error: {str(e)}",
                error=True
            )

    async def handle_sources(self, arguments: Dict[str, Any]) -> ToolResponse:
        """Handle sources tool call."""
        try:
            sources = self.db.get_all_sources()
            
            if not sources:
                return ToolResponse(
                    success=True,
                    content="No sources indexed yet. Use 'smurf' to index content."
                )
            
            message = f"📚 Indexed Sources ({len(sources)} total):\n\n"
            
            for source in sources:
                message += f"🔗 **{source.get('source_id', 'Unknown URL')}**\n"
                if source.get('summary'):
                    message += f"📝 {source['summary']}\n"
                if source.get('processor_name'):
                    message += f"📄 Type: {source['processor_name']}\n"
                if source.get('created_at'):
                    message += f"📅 Created: {source['created_at']}\n"
                if source.get('total_word_count'):
                    message += f"📊 Words: {source['total_word_count']}\n"
                message += "\n"
            
            return ToolResponse(success=True, content=message)
            
        except Exception as e:
            logger.error(f"Error in sources handler: {e}")
            return ToolResponse(
                success=False,
                content=f"❌ Error retrieving sources: {str(e)}",
                error=True
            )

    def run(self, host: str = "0.0.0.0", port: int = 8090):
        """Run the HTTP MCP server."""
        logger.info(f"🎮 Starting smurf HTTP MCP server on {host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  - GET  /health - Health check")
        logger.info("  - GET  /mcp/tools - List available tools")
        logger.info("  - POST /mcp/tools/call - Call a tool")
        logger.info("Tools: smurf, smurf_search, smurf_sources")
        
        uvicorn.run(
            self.app, 
            host=host, 
            port=port,
            log_level="info"
        )


def main():
    """Main entry point for HTTP MCP server."""
    server = SmurfHTTPMCPServer()
    
    # Get configuration from environment
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8090"))
    
    server.run(host=host, port=port)


if __name__ == "__main__":
    main() 