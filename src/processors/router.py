"""
Processor router for SNARF.
Routes URLs to appropriate processors and manages processing workflow.
"""
from typing import List, Dict, Any, Optional
import asyncio
from .base import BaseProcessor, ProcessorResult


class ProcessorRouter:
    """
    Routes URLs to appropriate processors and manages the processing workflow.
    Implements fallback logic and processor selection strategies.
    """
    
    def __init__(self, processors: List[BaseProcessor], default_processor: Optional[BaseProcessor] = None):
        """
        Initialize the router with a list of processors.
        
        Args:
            processors: List of processor instances
            default_processor: Fallback processor (typically web processor)
        """
        self.processors = processors
        self.default_processor = default_processor or (processors[0] if processors else None)
        self.stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'fallback_uses': 0,
            'processor_usage': {}
        }
    
    async def route(self, url: str, options: Dict[str, Any] = None) -> ProcessorResult:
        """
        Route a URL to the appropriate processor and process it.
        
        Args:
            url: The URL to process
            options: Optional processing options
            
        Returns:
            ProcessorResult from the selected processor
        """
        options = options or {}
        self.stats['total_requests'] += 1
        
        # Find the first processor that can handle this URL
        selected_processor = await self._select_processor(url, options)
        
        if selected_processor:
            processor_name = selected_processor.name
            self.stats['processor_usage'][processor_name] = self.stats['processor_usage'].get(processor_name, 0) + 1
            
            try:
                result = await selected_processor.process(url, options)
                if result.success:
                    self.stats['successful_routes'] += 1
                return result
            except Exception as e:
                print(f"Error processing with {processor_name}: {e}")
                # Fall back to default processor if the selected one fails
                if selected_processor != self.default_processor and self.default_processor:
                    return await self._fallback_process(url, options, str(e))
                else:
                    return ProcessorResult(
                        urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                        url_to_full_document={}, success=False, 
                        message=f"Processing failed: {e}",
                        processor_name=processor_name
                    )
        else:
            # No processor found, use default
            return await self._fallback_process(url, options, "No suitable processor found")
    
    async def _select_processor(self, url: str, options: Dict[str, Any]) -> Optional[BaseProcessor]:
        """
        Select the most appropriate processor for the given URL.
        
        Args:
            url: The URL to process
            options: Processing options
            
        Returns:
            Selected processor or None if no suitable processor found
        """
        # Check each processor in order of registration
        for processor in self.processors:
            if not processor.is_enabled():
                continue
                
            try:
                if await processor.can_handle(url, options):
                    return processor
            except Exception as e:
                print(f"Error checking if {processor.name} can handle {url}: {e}")
                continue
        
        return None
    
    async def _fallback_process(self, url: str, options: Dict[str, Any], reason: str) -> ProcessorResult:
        """
        Process URL with the default processor as fallback.
        
        Args:
            url: The URL to process
            options: Processing options
            reason: Reason for fallback
            
        Returns:
            ProcessorResult from default processor
        """
        self.stats['fallback_uses'] += 1
        
        if not self.default_processor:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"No default processor available. Reason: {reason}",
                processor_name="none"
            )
        
        try:
            print(f"Falling back to {self.default_processor.name} processor. Reason: {reason}")
            result = await self.default_processor.process(url, options)
            return result
        except Exception as e:
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"Fallback processing failed: {e}. Original reason: {reason}",
                processor_name=self.default_processor.name
            )
    
    async def process_multiple(self, urls: List[str], options: Dict[str, Any] = None) -> List[ProcessorResult]:
        """
        Process multiple URLs concurrently.
        
        Args:
            urls: List of URLs to process
            options: Optional processing options
            
        Returns:
            List of ProcessorResults
        """
        options = options or {}
        
        # Process URLs concurrently
        tasks = [self.route(url, options.copy()) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message=f"Exception during processing: {result}",
                    processor_name="unknown"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_processor_by_name(self, name: str) -> Optional[BaseProcessor]:
        """
        Get a processor by name.
        
        Args:
            name: Name of the processor
            
        Returns:
            Processor instance or None if not found
        """
        for processor in self.processors:
            if processor.name.lower() == name.lower():
                return processor
        return None
    
    def list_processors(self) -> List[Dict[str, Any]]:
        """
        List all registered processors with their status.
        
        Returns:
            List of processor information dictionaries
        """
        return [
            {
                'name': proc.name,
                'type': proc.processor_type,
                'enabled': proc.is_enabled(),
                'usage_count': self.stats['processor_usage'].get(proc.name, 0)
            }
            for proc in self.processors
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get router statistics.
        
        Returns:
            Dictionary with routing statistics
        """
        return {
            **self.stats,
            'success_rate': self.stats['successful_routes'] / max(1, self.stats['total_requests']),
            'fallback_rate': self.stats['fallback_uses'] / max(1, self.stats['total_requests']),
            'active_processors': len([p for p in self.processors if p.is_enabled()])
        }
    
    def reset_stats(self):
        """Reset router statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'fallback_uses': 0,
            'processor_usage': {}
        }
    
    def add_processor(self, processor: BaseProcessor):
        """
        Add a new processor to the router.
        
        Args:
            processor: Processor instance to add
        """
        if processor not in self.processors:
            self.processors.append(processor)
            self.stats['processor_usage'][processor.name] = 0
    
    def remove_processor(self, processor_name: str) -> bool:
        """
        Remove a processor from the router.
        
        Args:
            processor_name: Name of processor to remove
            
        Returns:
            True if processor was removed, False if not found
        """
        for i, processor in enumerate(self.processors):
            if processor.name.lower() == processor_name.lower():
                removed_processor = self.processors.pop(i)
                if self.default_processor == removed_processor:
                    self.default_processor = self.processors[0] if self.processors else None
                return True
        return False
    
    def set_default_processor(self, processor_name: str) -> bool:
        """
        Set the default fallback processor.
        
        Args:
            processor_name: Name of processor to set as default
            
        Returns:
            True if successful, False if processor not found
        """
        processor = self.get_processor_by_name(processor_name)
        if processor:
            self.default_processor = processor
            return True
        return False


class ProcessorPipeline:
    """
    Pipeline for chaining processors or applying transformations.
    Useful for complex processing workflows.
    """
    
    def __init__(self, stages: List[BaseProcessor]):
        """
        Initialize pipeline with processing stages.
        
        Args:
            stages: List of processors to apply in sequence
        """
        self.stages = stages
    
    async def process(self, url: str, options: Dict[str, Any] = None) -> ProcessorResult:
        """
        Process URL through all pipeline stages.
        
        Args:
            url: URL to process
            options: Processing options
            
        Returns:
            Final ProcessorResult after all stages
        """
        current_result = None
        options = options or {}
        
        for stage in self.stages:
            if not stage.is_enabled():
                continue
                
            try:
                if current_result is None:
                    # First stage processes the original URL
                    current_result = await stage.process(url, options)
                else:
                    # Subsequent stages can process the result from previous stage
                    # This allows for transformation pipelines
                    if hasattr(stage, 'process_result'):
                        current_result = await stage.process_result(current_result, options)
                    else:
                        # If stage doesn't support result processing, skip it
                        continue
                
                if not current_result.success:
                    break
                    
            except Exception as e:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message=f"Pipeline stage {stage.name} failed: {e}",
                    processor_name=stage.name
                )
        
        return current_result or ProcessorResult(
            urls=[], chunk_numbers=[], contents=[], metadatas={}, 
            url_to_full_document={}, success=False, 
            message="No pipeline stages executed",
            processor_name="pipeline"
        )