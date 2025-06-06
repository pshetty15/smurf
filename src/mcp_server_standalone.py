#!/usr/bin/env python3
"""
SNARF Standalone MCP Server
A generic MCP server that can be accessed by any MCP-compatible client
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, ImageContent, EmbeddedResource,
    ListToolsResult, CallToolResult, ErrorData
)

from dotenv import load_dotenv
from core.database import Database
from processors.web import WebProcessor
from processors.github import GitHubProcessor
from processors.router import ProcessorRouter


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SNARFMCPServer:
    """Standalone MCP Server for SNARF."""
    
    def __init__(self):
        self.db = None
        self.router = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize SNARF components."""
        if self.initialized:
            return
            
        try:
            logger.info("🚀 Initializing SNARF MCP server...")
            
            # Initialize database
            self.db = Database()
            await self.db.initialize()
            
            # Initialize processors
            web_processor = WebProcessor(self.db)
            github_processor = GitHubProcessor(self.db)
            
            # Create router
            self.router = ProcessorRouter([github_processor, web_processor], web_processor)
            
            self.initialized = True
            logger.info("✅ SNARF MCP server initialized!")
            
        except Exception as e:
            logger.error(f"Failed to initialize SNARF: {e}")
            raise
    
    async def handle_list_tools(self) -> ListToolsResult:
        """List available SNARF tools."""
        return ListToolsResult(
            tools=[
                Tool(
                    name="snarf",
                    description="""Crawl and index content from URLs into the SNARF knowledge base.
                    Supports:
                    - Web documentation (HTML pages)
                    - GitHub repositories
                    - Sitemaps (.xml)
                    - URL lists (.txt files)
                    
                    Examples:
                    - snarf https://docs.python.org/3/tutorial/
                    - snarf https://github.com/fastapi/fastapi
                    - snarf https://ai-sdk.dev/llms.txt
                    """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to crawl and index"
                            },
                            "smart_crawl": {
                                "type": "boolean",
                                "description": "Enable smart crawling for sitemaps/txt files",
                                "default": False
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Crawl depth for recursive crawling",
                                "default": 1
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="snarf_search",
                    description="""Search the SNARF knowledge base using semantic search.
                    Returns relevant chunks from indexed content.""",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            },
                            "source_filter": {
                                "type": "string",
                                "description": "Filter results by source domain"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="snarf_sources",
                    description="List all indexed sources in the SNARF knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        )
    
    async def handle_call_tool(self, name: str, arguments: Optional[Dict[str, Any]]) -> CallToolResult:
        """Handle tool calls."""
        try:
            if not self.initialized:
                await self.initialize()
            
            if name == "snarf":
                return await self.handle_snarf(arguments)
            elif name == "snarf_search":
                return await self.handle_search(arguments)
            elif name == "snarf_sources":
                return await self.handle_sources()
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return CallToolResult(
                content=[TextContent(text=f"Error: {str(e)}")],
                isError=True
            )
    
    async def handle_snarf(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle snarf tool call for crawling."""
        url = arguments.get("url")
        if not url:
            return CallToolResult(
                content=[TextContent(text="Error: URL is required")],
                isError=True
            )
        
        try:
            # Detect URL type for smart handling
            parsed = urlparse(url)
            is_txt_file = url.endswith('.txt')
            is_sitemap = 'sitemap' in url.lower() or url.endswith('.xml')
            
            options = {
                "smart_crawl": arguments.get("smart_crawl", is_txt_file or is_sitemap),
                "depth": arguments.get("depth", 1)
            }
            
            logger.info(f"🌐 MCP: Processing URL: {url}")
            
            # Process the URL
            result = await self.router.route(url, options)
            
            if result.success:
                # Format success message
                message = f"✅ Successfully indexed {len(result.contents)} chunks from {url}\n"
                message += f"📊 Processor: {result.processor_name}\n"
                
                # Add specific details based on processor
                if result.processor_name == "github":
                    # Extract repository info
                    repo_parts = parsed.path.strip('/').split('/')
                    if len(repo_parts) >= 2:
                        message += f"📦 Repository: {repo_parts[0]}/{repo_parts[1]}\n"
                
                elif result.processor_name == "web" and options["smart_crawl"]:
                    # Count unique URLs processed
                    unique_urls = len(set(result.urls))
                    message += f"📄 Pages processed: {unique_urls}\n"
                
                message += f"\nUse 'snarf_search' to query the indexed content."
                
                return CallToolResult(content=[TextContent(text=message)])
            else:
                return CallToolResult(
                    content=[TextContent(text=f"❌ Failed to index {url}: {result.message}")],
                    isError=True
                )
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return CallToolResult(
                content=[TextContent(text=f"Error processing URL: {str(e)}")],
                isError=True
            )
    
    async def handle_search(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle search tool call."""
        query = arguments.get("query")
        if not query:
            return CallToolResult(
                content=[TextContent(text="Error: Query is required")],
                isError=True
            )
        
        try:
            limit = arguments.get("limit", 10)
            source_filter = arguments.get("source_filter")
            
            logger.info(f"🔍 MCP: Searching for: {query}")
            
            # Search the database
            results = self.db.search_documents(query, limit, source_filter=source_filter)
            
            if not results:
                return CallToolResult(
                    content=[TextContent(text=f"No results found for: {query}")]
                )
            
            # Format results
            message = f"🔍 Search results for '{query}' ({len(results)} found):\n\n"
            
            for i, result in enumerate(results, 1):
                similarity = result.get('similarity', 0)
                url = result.get('url', 'Unknown')
                content_preview = result.get('content', '')[:200].replace('\n', ' ')
                source = result.get('source_id', 'Unknown')
                
                message += f"{i}. [{similarity:.3f}] {url}\n"
                message += f"   Source: {source}\n"
                message += f"   Preview: {content_preview}...\n\n"
            
            return CallToolResult(content=[TextContent(text=message)])
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return CallToolResult(
                content=[TextContent(text=f"Search error: {str(e)}")],
                isError=True
            )
    
    async def handle_sources(self) -> CallToolResult:
        """Handle sources listing."""
        try:
            sources = self.db.get_all_sources()
            
            if not sources:
                return CallToolResult(
                    content=[TextContent(text="No sources indexed yet. Use 'snarf' to index content.")]
                )
            
            # Format sources
            message = f"📚 Indexed sources ({len(sources)} total):\n\n"
            
            for source in sources:
                source_id = source.get('source_id', 'Unknown')
                summary = source.get('summary', 'No summary')[:100]
                word_count = source.get('total_word_count', 0)
                processor = source.get('processor_name', 'unknown')
                
                message += f"• {source_id}\n"
                message += f"  Processor: {processor}\n"
                message += f"  Word count: {word_count:,}\n"
                message += f"  Summary: {summary}...\n\n"
            
            return CallToolResult(content=[TextContent(text=message)])
            
        except Exception as e:
            logger.error(f"Sources listing error: {e}")
            return CallToolResult(
                content=[TextContent(text=f"Error listing sources: {str(e)}")],
                isError=True
            )


async def main():
    """Run the MCP server using stdio transport."""
    server = Server("snarf")
    snarf = SNARFMCPServer()
    
    # Initialize SNARF
    await snarf.initialize()
    
    # Register handlers
    @server.list_tools()
    async def handle_list_tools() -> ListToolsResult:
        return await snarf.handle_list_tools()
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> CallToolResult:
        return await snarf.handle_call_tool(name, arguments or {})
    
    # Run the server
    logger.info("🎮 SNARF MCP server running on stdio...")
    logger.info("Available tools: snarf, snarf_search, snarf_sources")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 SNARF MCP server shutdown")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise