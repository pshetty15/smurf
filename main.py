#!/usr/bin/env python3
"""
smurf - smart neural universal retrieval framework
Main application entry point for containerized deployment
"""

import asyncio
import os
import signal
import sys
import logging
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
        logging.FileHandler('/app/logs/smurf.log') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class SmurfApplication:
    """Main smurf application class."""
    
    def __init__(self):
        self.db = None
        self.router = None
        self.running = False
        
    async def initialize(self):
        """Initialize the application components."""
        try:
            logger.info("🚀 Initializing smurf application...")
            
            # Load environment variables
            load_dotenv()
            
            # Check required environment variables
            required_vars = ['AWS_PROFILE', 'AWS_REGION']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
            
            # Initialize database
            logger.info("📊 Connecting to database...")
            self.db = Database()
            await self.db.initialize()
            
            # Initialize processors
            logger.info("🔧 Setting up processors...")
            web_processor = WebProcessor(self.db)
            github_processor = GitHubProcessor(self.db)
            
            # Create router with processors (GitHub first for better matching)
            self.router = ProcessorRouter([github_processor, web_processor], web_processor)
            
            logger.info("✅ smurf initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize smurf: {e}")
            raise
    
    async def process_url(self, url: str, options: dict = None):
        """Process a single URL."""
        if not self.router:
            raise RuntimeError("Application not initialized")
        
        logger.info(f"🌐 Processing URL: {url}")
        result = await self.router.route(url, options or {})
        
        if result.success:
            logger.info(f"✅ Successfully processed {len(result.contents)} chunks from {url}")
            return {
                'success': True,
                'chunks': len(result.contents),
                'message': result.message,
                'url': url
            }
        else:
            logger.error(f"❌ Failed to process {url}: {result.message}")
            return {
                'success': False,
                'message': result.message,
                'url': url
            }
    
    async def search(self, query: str, limit: int = 10, source_filter: str = None):
        """Search the knowledge base."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        
        logger.info(f"🔍 Searching for: {query}")
        results = self.db.search_documents(query, limit, source_filter=source_filter)
        
        logger.info(f"📋 Found {len(results)} results")
        return {
            'query': query,
            'results': results,
            'count': len(results)
        }
    
    async def get_sources(self):
        """Get all available sources."""
        if not self.db:
            raise RuntimeError("Database not initialized")
        
        sources = self.db.get_all_sources()
        return {
            'sources': sources,
            'count': len(sources)
        }
    
    async def demo_mode(self):
        """Run in demo mode with sample URLs."""
        logger.info("🎮 Running smurf in demo mode...")
        
        demo_urls = [
            "https://docs.python.org/3/tutorial/introduction.html",
            "https://fastapi.tiangolo.com/tutorial/",
        ]
        
        for url in demo_urls:
            try:
                result = await self.process_url(url)
                logger.info(f"Demo result: {result}")
                await asyncio.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Demo processing error for {url}: {e}")
        
        # Demo search
        try:
            search_result = await self.search("python functions", limit=5)
            logger.info(f"Demo search results: {search_result['count']} found")
        except Exception as e:
            logger.error(f"Demo search error: {e}")
    
    async def run_interactive(self):
        """Run in interactive mode."""
        logger.info("💬 Starting smurf interactive mode...")
        logger.info("Commands: 'smurf <url>', 'search <query>', 'sources', 'quit'")
        
        self.running = True
        
        while self.running:
            try:
                command = input("\nsmurf> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.startswith('smurf '):
                    url = command[6:].strip()
                    result = await self.process_url(url)
                    print(f"Result: {result}")
                elif command.startswith('search '):
                    query = command[7:].strip()
                    result = await self.search(query)
                    print(f"Found {result['count']} results:")
                    for i, r in enumerate(result['results'][:3], 1):
                        print(f"  {i}. {r['url']} (similarity: {r.get('similarity', 0):.3f})")
                elif command == 'sources':
                    result = await self.get_sources()
                    print(f"Available sources ({result['count']}):")
                    for source in result['sources'][:5]:
                        print(f"  - {source['source_id']}: {source.get('summary', 'No summary')[:100]}...")
                elif command == 'help':
                    print("Commands:")
                    print("  smurf <url>     - Process a URL")
                    print("  search <query>  - Search the knowledge base")
                    print("  sources         - List available sources")
                    print("  quit            - Exit smurf")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Cleaning up smurf resources...")
        
        if self.router:
            for processor in self.router.processors:
                if hasattr(processor, 'cleanup'):
                    try:
                        await processor.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up processor {processor.name}: {e}")
        
        logger.info("✅ Cleanup complete")


async def main():
    """Main entry point."""
    app = SmurfApplication()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        app.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize application
        await app.initialize()
        
        # Parse mode from environment
        mode = os.getenv("SMURF_MODE", "interactive")
        
        if mode == "demo":
            await app.demo_mode()
        elif mode == "api":
            # API mode is handled by separate api_server.py
            logger.info("API mode requires running api_server.py separately")
            await app.run_interactive()
        else:
            await app.run_interactive()
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        await app.cleanup()
        logger.info("👋 smurf shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 smurf shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)