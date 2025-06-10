#!/usr/bin/env python3
"""
smurf Standalone MCP Server
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

from mcp.server import Server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)
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


class SmurfMCPServer:
    """Standalone MCP Server for smurf."""
    
    def __init__(self):
        self.db = None
        self.router = None
        self.initialized = False

    async def initialize(self):
        """Initialize smurf components."""
        if self.initialized:
            return
            
        try:
            logger.info("🚀 Initializing smurf MCP server...")
            
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
            logger.info("✅ smurf MCP server initialized!")
            
        except Exception as e:
            logger.error(f"Failed to initialize smurf: {e}")
            raise

    async def handle_list_tools(self) -> ListToolsResult:
        """List available smurf tools."""
        tools = [
            Tool(
                name="smurf",
                description="""Crawl and index content from URLs into the smurf knowledge base.
                
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
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to crawl and index"
                        }
                    },
                    "required": ["url"]
                }
            ),
            Tool(
                name="smurf_search",
                description="""Search the smurf knowledge base using semantic search.
                
Uses vector embeddings to find semantically similar content.
Returns ranked results with similarity scores.

Examples:
- smurf_search "machine learning tutorials"
- smurf_search "FastAPI authentication"
- smurf_search "Python async programming"

Returns: List of relevant content chunks with URLs and scores""",
                inputSchema={
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
            ),
            Tool(
                name="smurf_sources",
                description="List all indexed sources in the smurf knowledge base",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
        
        return ListToolsResult(tools=tools)

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle tool calls."""
        if not self.initialized:
            await self.initialize()
            
        if name == "smurf":
            return await self.handle_smurf(arguments)
        elif name == "smurf_search":
            return await self.handle_search(arguments)
        elif name == "smurf_sources":
            return await self.handle_sources(arguments)
        else:
            return CallToolResult(
                content=[TextContent(text=f"Unknown tool: {name}")],
                isError=True
            )

    async def handle_smurf(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle smurf tool call for crawling."""
        try:
            url = arguments.get("url")
            if not url:
                return CallToolResult(
                    content=[TextContent(text="Error: URL is required")],
                    isError=True
                )
            
            # Process the URL
            result = await self.router.process_url(url)
            
            if result['success']:
                message = f"✅ Successfully indexed content from: {url}\n"
                message += f"📊 Processed {result.get('chunks', 0)} content chunks\n"
                
                if result.get('details'):
                    details = result['details']
                    if details.get('title'):
                        message += f"📝 Title: {details['title']}\n"
                    if details.get('content_type'):
                        message += f"📄 Type: {details['content_type']}\n"
                    if details.get('file_count'):
                        message += f"📁 Files: {details['file_count']}\n"
                    if details.get('commit_count'):
                        message += f"🔄 Commits: {details['commit_count']}\n"
                
                message += f"\nUse 'smurf_search' to query the indexed content."
                
                return CallToolResult(
                    content=[TextContent(text=message)]
                )
            else:
                return CallToolResult(
                    content=[TextContent(text=f"❌ Failed to process {url}: {result.get('error', 'Unknown error')}")],
                    isError=True
                )
                
        except Exception as e:
            logger.error(f"Error in smurf handler: {e}")
            return CallToolResult(
                content=[TextContent(text=f"❌ Error processing URL: {str(e)}")],
                isError=True
            )

    async def handle_search(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle search tool call."""
        try:
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            
            if not query:
                return CallToolResult(
                    content=[TextContent(text="Error: Query is required")],
                    isError=True
                )
            
            # Perform search
            results = await self.db.search_chunks(query, limit=limit)
            
            if not results:
                return CallToolResult(
                    content=[TextContent(text=f"No results found for: {query}")],
                    isError=False
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
            
            return CallToolResult(
                content=[TextContent(text=message)]
            )
            
        except Exception as e:
            logger.error(f"Error in search handler: {e}")
            return CallToolResult(
                content=[TextContent(text=f"❌ Search error: {str(e)}")],
                isError=True
            )

    async def handle_sources(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle sources tool call."""
        try:
            sources = await self.db.get_all_sources()
            
            if not sources:
                return CallToolResult(
                    content=[TextContent(text="No sources indexed yet. Use 'smurf' to index content.")]
                )
            
            message = f"📚 Indexed Sources ({len(sources)} total):\n\n"
            
            for source in sources:
                message += f"🔗 **{source.get('url', 'Unknown URL')}**\n"
                if source.get('title'):
                    message += f"📝 {source['title']}\n"
                if source.get('content_type'):
                    message += f"📄 Type: {source['content_type']}\n"
                if source.get('indexed_at'):
                    message += f"📅 Indexed: {source['indexed_at']}\n"
                message += "\n"
            
            return CallToolResult(
                content=[TextContent(text=message)]
            )
            
        except Exception as e:
            logger.error(f"Error in sources handler: {e}")
            return CallToolResult(
                content=[TextContent(text=f"❌ Error retrieving sources: {str(e)}")],
                isError=True
            )


async def main():
    """Main entry point for standalone MCP server."""
    
    # Create server instance
    server = Server("smurf")
    smurf = SmurfMCPServer()
    
    # Initialize smurf
    await smurf.initialize()
    
    @server.list_tools()
    async def handle_list_tools() -> ListToolsResult:
        return await smurf.handle_list_tools()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any] | None) -> CallToolResult:
        return await smurf.handle_call_tool(name, arguments or {})

    logger.info("🎮 smurf MCP server running on stdio...")
    logger.info("Available tools: smurf, smurf_search, smurf_sources")
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        async with server.run_with_stdin_stdout() as server_process:
            await shutdown_event.wait()
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("👋 smurf MCP server shutdown")


if __name__ == "__main__":
    asyncio.run(main())