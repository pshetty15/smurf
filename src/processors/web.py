"""
Web processor for smurf.
Handles web crawling using Crawl4AI - extracted from crawl4ai_mcp.py
"""
import os
import asyncio
import json
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
import requests
import concurrent.futures

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from .base import BaseProcessor, ProcessorResult, URLMatcher, register_processor
from ..core.embeddings import extract_code_blocks, generate_code_example_summary, extract_source_summary


@register_processor
class WebProcessor(BaseProcessor):
    """
    Web content processor using Crawl4AI.
    Handles regular web pages, sitemaps, and text files.
    """
    
    def __init__(self, database):
        super().__init__(database, "web")
        self.crawler = None
        self._crawler_lock = asyncio.Lock()
    
    async def can_handle(self, url: str, options: Dict[str, Any] = None) -> bool:
        """Check if this is a web URL that we can crawl."""
        # Handle most HTTP/HTTPS URLs except GitHub repositories
        if URLMatcher.is_web_url(url):
            # Don't handle GitHub repositories - those go to GitHub processor
            if URLMatcher.is_github_url(url):
                return False
            return True
        return False
    
    async def process(self, url: str, options: Dict[str, Any] = None) -> ProcessorResult:
        """Process web content using Crawl4AI."""
        options = options or {}
        
        try:
            # Initialize crawler if needed
            await self._ensure_crawler()
            
            # Determine if this is a smart crawl (multiple pages) or single page
            is_smart_crawl = options.get('smart_crawl', False)
            max_depth = options.get('depth', 1)
            
            if is_smart_crawl:
                return await self._smart_crawl(url, max_depth)
            else:
                return await self._crawl_single_page(url)
                
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Web processing failed: {e}",
                processor_name=self.name
            )
    
    async def _ensure_crawler(self):
        """Ensure crawler is initialized."""
        async with self._crawler_lock:
            if self.crawler is None:
                browser_config = BrowserConfig(headless=True, verbose=False)
                self.crawler = AsyncWebCrawler(config=browser_config)
                await self.crawler.__aenter__()
    
    async def _crawl_single_page(self, url: str) -> ProcessorResult:
        """Crawl a single web page."""
        try:
            # Create simple crawler configuration without extraction strategy
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=1
            )
            
            # Crawl the page
            result = await self.crawler.arun(url=url, config=config)
            
            if not result.success:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message=f"Crawling failed: {result.error_message}",
                    processor_name=self.name
                )
            
            # Process the content
            markdown_content = result.markdown or ""
            if not markdown_content.strip():
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No content extracted from page",
                    processor_name=self.name
                )
            
            # Chunk the content
            chunks = self._chunk_markdown_content(markdown_content, url)
            
            if not chunks:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No chunks created from content",
                    processor_name=self.name
                )
            
            # Prepare result data
            urls = [chunk['url'] for chunk in chunks]
            chunk_numbers = [chunk['chunk_number'] for chunk in chunks]
            contents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            url_to_full_document = {url: markdown_content}
            
            # Store in database
            source_id = self.get_source_id(url)
            summary = extract_source_summary(source_id, markdown_content[:5000])
            word_count = len(markdown_content.split())
            
            self.database.update_source_info(source_id, summary, word_count, self.name)
            self.database.store_documents(
                urls, chunk_numbers, contents, metadatas, 
                url_to_full_document, self.name
            )
            
            # Process code examples if enabled
            if os.getenv("USE_AGENTIC_RAG", "false") == "true":
                await self._process_code_examples(url, markdown_content)
            
            return ProcessorResult(
                urls=urls, chunk_numbers=chunk_numbers, contents=contents, 
                metadatas=metadatas, url_to_full_document=url_to_full_document, 
                success=True, message=f"Successfully crawled {len(chunks)} chunks",
                processor_name=self.name
            )
            
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Single page crawl failed: {e}",
                processor_name=self.name
            )
    
    async def _smart_crawl(self, url: str, max_depth: int = 1) -> ProcessorResult:
        """Intelligently crawl based on URL type (sitemap, txt file, or recursive)."""
        try:
            # Detect URL type and route accordingly
            if self._is_sitemap_url(url):
                return await self._crawl_sitemap(url)
            elif self._is_txt_file_url(url):
                return await self._crawl_txt_file(url)
            else:
                return await self._crawl_recursive(url, max_depth)
                
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Smart crawl failed: {e}",
                processor_name=self.name
            )
    
    def _is_sitemap_url(self, url: str) -> bool:
        """Check if URL is a sitemap."""
        return 'sitemap' in url.lower() or url.endswith('.xml')
    
    def _is_txt_file_url(self, url: str) -> bool:
        """Check if URL is a text file listing."""
        return url.endswith('.txt') or 'llms-full.txt' in url
    
    async def _crawl_sitemap(self, sitemap_url: str) -> ProcessorResult:
        """Crawl pages from a sitemap."""
        try:
            # Download and parse sitemap
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # Parse XML sitemap
            root = ElementTree.fromstring(response.content)
            urls = []
            
            # Handle different sitemap formats
            for elem in root.iter():
                if elem.tag.endswith('loc'):
                    page_url = elem.text
                    if page_url and page_url.startswith('http'):
                        urls.append(page_url)
            
            if not urls:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No URLs found in sitemap",
                    processor_name=self.name
                )
            
            # Crawl pages in parallel
            return await self._crawl_multiple_pages(urls[:50])  # Limit to 50 pages
            
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Sitemap crawl failed: {e}",
                processor_name=self.name
            )
    
    async def _crawl_txt_file(self, txt_url: str) -> ProcessorResult:
        """Crawl pages from a text file listing."""
        try:
            # Download text file
            response = requests.get(txt_url, timeout=30)
            response.raise_for_status()
            
            # Parse URLs from text file
            urls = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and line.startswith('http'):
                    urls.append(line)
            
            if not urls:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No URLs found in text file",
                    processor_name=self.name
                )
            
            # Crawl pages in parallel
            return await self._crawl_multiple_pages(urls[:50])  # Limit to 50 pages
            
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Text file crawl failed: {e}",
                processor_name=self.name
            )
    
    async def _crawl_recursive(self, start_url: str, max_depth: int) -> ProcessorResult:
        """Recursively crawl a website."""
        try:
            visited_urls = set()
            urls_to_visit = [(start_url, 0)]  # (url, depth)
            all_results = []
            
            base_domain = urlparse(start_url).netloc
            
            while urls_to_visit and len(visited_urls) < 50:  # Limit total pages
                current_url, depth = urls_to_visit.pop(0)
                
                if current_url in visited_urls or depth > max_depth:
                    continue
                
                visited_urls.add(current_url)
                
                # Crawl current page
                result = await self._crawl_single_page(current_url)
                if result.success:
                    all_results.append(result)
                    
                    # Find links on this page for next depth level
                    if depth < max_depth:
                        # This would require parsing the HTML for links
                        # Simplified implementation - in practice you'd extract links
                        pass
            
            # Combine all results
            if all_results:
                combined = self._combine_results(all_results)
                return combined
            else:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No pages successfully crawled",
                    processor_name=self.name
                )
                
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Recursive crawl failed: {e}",
                processor_name=self.name
            )
    
    async def _crawl_multiple_pages(self, urls: List[str]) -> ProcessorResult:
        """Crawl multiple pages in parallel."""
        try:
            # Process pages in smaller batches to avoid overwhelming the server
            batch_size = 10
            all_results = []
            
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                
                # Create crawler configurations for each URL
                configs = [
                    CrawlerRunConfig(
                        cache_mode=CacheMode.BYPASS,
                        word_count_threshold=1
                    ) for _ in batch_urls
                ]
                
                # Crawl batch in parallel
                batch_results = await asyncio.gather(
                    *[self.crawler.arun(url=url, config=config) 
                      for url, config in zip(batch_urls, configs)],
                    return_exceptions=True
                )
                
                # Process results
                for url, result in zip(batch_urls, batch_results):
                    if isinstance(result, Exception):
                        print(f"Error crawling {url}: {result}")
                        continue
                    
                    if result.success and result.markdown:
                        page_result = await self._process_page_result(url, result.markdown)
                        if page_result.success:
                            all_results.append(page_result)
            
            # Combine all results
            if all_results:
                return self._combine_results(all_results)
            else:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No pages successfully processed",
                    processor_name=self.name
                )
                
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Multiple page crawl failed: {e}",
                processor_name=self.name
            )
    
    async def _process_page_result(self, url: str, markdown_content: str) -> ProcessorResult:
        """Process a single page result into chunks."""
        try:
            chunks = self._chunk_markdown_content(markdown_content, url)
            if not chunks:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message=f"No chunks created for {url}",
                    processor_name=self.name
                )
            
            urls = [chunk['url'] for chunk in chunks]
            chunk_numbers = [chunk['chunk_number'] for chunk in chunks]
            contents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            url_to_full_document = {url: markdown_content}
            
            return ProcessorResult(
                urls=urls, chunk_numbers=chunk_numbers, contents=contents, 
                metadatas=metadatas, url_to_full_document=url_to_full_document, 
                success=True, message=f"Processed {len(chunks)} chunks from {url}",
                processor_name=self.name
            )
            
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Page processing failed for {url}: {e}",
                processor_name=self.name
            )
    
    def _chunk_markdown_content(self, content: str, url: str) -> List[Dict[str, Any]]:
        """Chunk markdown content by headers and size."""
        chunks = []
        
        # Split by headers first
        sections = re.split(r'\n(#{1,6}\s+.*)', content)
        
        current_chunk = ""
        chunk_number = 0
        
        for i, section in enumerate(sections):
            if i == 0:
                # First section before any headers
                current_chunk = section.strip()
            elif section.startswith('#'):
                # This is a header - start new chunk with previous content
                if current_chunk.strip():
                    chunks.append({
                        'url': url,
                        'chunk_number': chunk_number,
                        'content': current_chunk.strip(),
                        'metadata': {
                            'chunk_size': len(current_chunk.strip()),
                            'processor': self.name,
                            'content_type': 'documentation'
                        }
                    })
                    chunk_number += 1
                
                current_chunk = section
            else:
                # Content under current header
                current_chunk += "\n" + section
                
                # If chunk is getting too large, split it
                if len(current_chunk) > 2000:
                    chunks.append({
                        'url': url,
                        'chunk_number': chunk_number,
                        'content': current_chunk.strip(),
                        'metadata': {
                            'chunk_size': len(current_chunk.strip()),
                            'processor': self.name,
                            'content_type': 'documentation'
                        }
                    })
                    chunk_number += 1
                    current_chunk = ""
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'url': url,
                'chunk_number': chunk_number,
                'content': current_chunk.strip(),
                'metadata': {
                    'chunk_size': len(current_chunk.strip()),
                    'processor': self.name,
                    'content_type': 'documentation'
                }
            })
        
        return chunks
    
    async def _process_code_examples(self, url: str, content: str):
        """Extract and process code examples from content."""
        try:
            code_blocks = extract_code_blocks(content, min_length=300)
            
            if not code_blocks:
                return
            
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            for i, block in enumerate(code_blocks):
                # Generate summary for code block
                summary = generate_code_example_summary(
                    block['code'], 
                    block['context_before'], 
                    block['context_after']
                )
                
                code_urls.append(url)
                code_chunk_numbers.append(i)
                code_examples.append(block['code'])
                code_summaries.append(summary)
                code_metadatas.append({
                    'language': block['language'],
                    'context_before': block['context_before'][:200],
                    'context_after': block['context_after'][:200],
                    'word_count': len(block['code'].split()),
                    'processor': self.name
                })
            
            # Store code examples
            self.database.store_code_examples(
                code_urls, code_chunk_numbers, code_examples, 
                code_summaries, code_metadatas, self.name
            )
            
        except Exception as e:
            print(f"Error processing code examples: {e}")
    
    def _combine_results(self, results: List[ProcessorResult]) -> ProcessorResult:
        """Combine multiple processor results into one."""
        all_urls = []
        all_chunk_numbers = []
        all_contents = []
        all_metadatas = []
        all_url_to_full_document = {}
        
        for result in results:
            if result.success:
                all_urls.extend(result.urls)
                all_chunk_numbers.extend(result.chunk_numbers)
                all_contents.extend(result.contents)
                all_metadatas.extend(result.metadatas)
                all_url_to_full_document.update(result.url_to_full_document)
        
        return ProcessorResult(
            urls=all_urls, chunk_numbers=all_chunk_numbers, contents=all_contents, 
            metadatas=all_metadatas, url_to_full_document=all_url_to_full_document, 
            success=len(all_urls) > 0, 
            message=f"Combined {len(results)} results into {len(all_urls)} total chunks",
            processor_name=self.name
        )
    
    async def cleanup(self):
        """Clean up crawler resources."""
        if self.crawler:
            await self.crawler.__aexit__(None, None, None)
            self.crawler = None