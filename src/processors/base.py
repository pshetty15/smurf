"""
Base processor interface for SNARF.
All source processors inherit from this base class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import re


class ProcessorResult:
    """Result from processing a source."""
    
    def __init__(
        self, 
        urls: List[str], 
        chunk_numbers: List[int],
        contents: List[str], 
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        success: bool = True,
        message: str = "",
        processor_name: str = "unknown"
    ):
        self.urls = urls
        self.chunk_numbers = chunk_numbers
        self.contents = contents
        self.metadatas = metadatas
        self.url_to_full_document = url_to_full_document
        self.success = success
        self.message = message
        self.processor_name = processor_name
        
        # Validation
        if success and len(urls) != len(contents):
            raise ValueError("URLs and contents must have same length")
        if success and len(contents) != len(metadatas):
            raise ValueError("Contents and metadatas must have same length")


class BaseProcessor(ABC):
    """
    Abstract base class for all source processors.
    Each processor handles a specific type of source (web, GitHub, SonarQube, etc.)
    """
    
    def __init__(self, database, name: str):
        """
        Initialize the processor.
        
        Args:
            database: Database instance for storage
            name: Name of this processor
        """
        self.database = database
        self.name = name
        self.enabled = True
    
    @abstractmethod
    async def can_handle(self, url: str, options: Dict[str, Any] = None) -> bool:
        """
        Check if this processor can handle the given URL.
        
        Args:
            url: The URL to check
            options: Optional processing options
            
        Returns:
            True if this processor can handle the URL
        """
        pass
    
    @abstractmethod
    async def process(self, url: str, options: Dict[str, Any] = None) -> ProcessorResult:
        """
        Process the source and return structured data.
        
        Args:
            url: The URL to process
            options: Optional processing options
            
        Returns:
            ProcessorResult with chunks and metadata
        """
        pass
    
    def get_source_id(self, url: str) -> str:
        """
        Extract source ID from URL.
        Default implementation uses domain.
        
        Args:
            url: The URL to extract source ID from
            
        Returns:
            Source identifier
        """
        parsed_url = urlparse(url)
        return parsed_url.netloc or parsed_url.path
    
    def chunk_content(
        self, 
        content: str, 
        url: str,
        chunk_size: int = 1000, 
        overlap: int = 200,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Default content chunking strategy.
        Can be overridden by specific processors.
        
        Args:
            content: Content to chunk
            url: Source URL
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            **kwargs: Additional chunking options
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not content or len(content.strip()) == 0:
            return []
        
        chunks = []
        content = content.strip()
        
        # Simple sliding window chunking
        start = 0
        chunk_number = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at word boundaries
            if end < len(content):
                # Look for sentence boundary first
                sentence_break = content.rfind('.', start, end)
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break + 1
                else:
                    # Fall back to word boundary
                    word_break = content.rfind(' ', start, end)
                    if word_break > start + chunk_size // 2:
                        end = word_break
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append({
                    'url': url,
                    'chunk_number': chunk_number,
                    'content': chunk_text,
                    'metadata': {
                        'chunk_size': len(chunk_text),
                        'start_pos': start,
                        'end_pos': end,
                        'processor': self.name
                    }
                })
                chunk_number += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            if start >= end:
                break
        
        return chunks
    
    def is_enabled(self) -> bool:
        """Check if this processor is enabled."""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable this processor."""
        self.enabled = enabled
    
    @property
    def processor_type(self) -> str:
        """Return the processor type for categorization."""
        return self.name.lower()


class URLMatcher:
    """Helper class for URL pattern matching."""
    
    @staticmethod
    def matches_domain(url: str, domains: List[str]) -> bool:
        """Check if URL matches any of the given domains."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(d.lower() in domain for d in domains)
    
    @staticmethod
    def matches_pattern(url: str, patterns: List[str]) -> bool:
        """Check if URL matches any regex pattern."""
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in patterns)
    
    @staticmethod
    def is_github_url(url: str) -> bool:
        """Check if URL is a GitHub repository URL."""
        patterns = [
            r'github\.com/[^/]+/[^/]+',
            r'^https?://github\.com/[^/]+/[^/]+/?$',
            r'^git@github\.com:[^/]+/[^/]+\.git$'
        ]
        return URLMatcher.matches_pattern(url, patterns)
    
    @staticmethod
    def is_web_url(url: str) -> bool:
        """Check if URL is a regular web URL."""
        return url.startswith(('http://', 'https://'))
    
    @staticmethod
    def is_file_url(url: str) -> bool:
        """Check if URL points to a file."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) for ext in ['.txt', '.md', '.rst', '.pdf', '.xml'])


# Registry for processor types
PROCESSOR_REGISTRY = {}


def register_processor(processor_class):
    """Decorator to register a processor class."""
    PROCESSOR_REGISTRY[processor_class.__name__] = processor_class
    return processor_class


def get_registered_processors():
    """Get all registered processor classes."""
    return PROCESSOR_REGISTRY.copy()