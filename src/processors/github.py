"""
GitHub processor for SNARF.
Handles GitHub repository cloning, analysis, and code structure extraction.
"""

import os
import asyncio
import logging
import hashlib
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from .base import BaseProcessor, ProcessorResult, URLMatcher, register_processor
from ..utils.git_utils import repo_manager
from ..core.embeddings import create_embeddings_batch, generate_code_example_summary


logger = logging.getLogger(__name__)


@register_processor
class GitHubProcessor(BaseProcessor):
    """
    GitHub repository processor.
    Clones repositories, analyzes code structure, and creates embeddings.
    """
    
    def __init__(self, database):
        super().__init__(database, "github")
        self.supported_extensions = [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.sh', '.sql',
            '.html', '.css', '.scss', '.md', '.rst', '.txt', '.json', '.yaml', '.yml'
        ]
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB default
    
    async def can_handle(self, url: str, options: Dict[str, Any] = None) -> bool:
        """Check if this is a GitHub repository URL."""
        return URLMatcher.is_github_url(url)
    
    async def process(self, url: str, options: Dict[str, Any] = None) -> ProcessorResult:
        """Process a GitHub repository."""
        options = options or {}
        
        try:
            logger.info(f"🔧 Processing GitHub repository: {url}")
            
            # Get repository instance
            repo = await repo_manager.get_repository(url)
            
            # Clone or update repository
            repo_path = await repo.clone_or_update()
            
            # Get repository info
            repo_info = repo.get_repository_info()
            
            # Update repository record in database
            await self._update_repository_record(repo, repo_info)
            
            # Get file list
            files = repo.get_file_list(self.supported_extensions)
            if not files:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No supported files found in repository",
                    processor_name=self.name
                )
            
            logger.info(f"📁 Found {len(files)} files to process")
            
            # Process files in batches
            all_chunks = []
            batch_size = options.get('batch_size', 10)
            
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch_chunks = await self._process_file_batch(repo, batch_files, url)
                all_chunks.extend(batch_chunks)
                
                # Progress logging
                logger.info(f"📊 Processed {min(i + batch_size, len(files))}/{len(files)} files")
            
            if not all_chunks:
                return ProcessorResult(
                    urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                    url_to_full_document={}, success=False, 
                    message="No content chunks created",
                    processor_name=self.name
                )
            
            # Prepare result data
            urls = [chunk['url'] for chunk in all_chunks]
            chunk_numbers = [chunk['chunk_number'] for chunk in all_chunks]
            contents = [chunk['content'] for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]
            url_to_full_document = {chunk['url']: chunk['full_content'] for chunk in all_chunks}
            
            # Store in database
            source_id = repo.repo_id
            summary = await self._generate_repository_summary(repo_info, all_chunks[:5])
            total_lines = sum(chunk['metadata'].get('line_count', 0) for chunk in all_chunks)
            
            self.database.update_source_info(source_id, summary, total_lines, self.name)
            self.database.store_documents(
                urls, chunk_numbers, contents, metadatas, 
                url_to_full_document, self.name
            )
            
            logger.info(f"✅ Successfully processed {len(all_chunks)} chunks from {repo.repo_id}")
            
            return ProcessorResult(
                urls=urls, chunk_numbers=chunk_numbers, contents=contents, 
                metadatas=metadatas, url_to_full_document=url_to_full_document, 
                success=True, 
                message=f"Successfully processed {len(all_chunks)} chunks from {len(files)} files",
                processor_name=self.name
            )
            
        except Exception as e:
            logger.error(f"❌ GitHub processing failed for {url}: {e}")
            return ProcessorResult(
                urls=[], chunk_numbers=[], contents=[], metadatas={}, 
                url_to_full_document={}, success=False, 
                message=f"GitHub processing failed: {e}",
                processor_name=self.name
            )
    
    async def _update_repository_record(self, repo, repo_info: Dict[str, Any]):
        """Update repository record in database."""
        try:
            conn = self.database.get_connection()
            cursor = conn.cursor()
            
            # Insert or update repository record
            cursor.execute("""
                INSERT INTO repositories (
                    repo_id, url, name, owner, description, 
                    last_indexed, index_status, repository_stats
                ) VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
                ON CONFLICT (repo_id) DO UPDATE SET
                    last_indexed = NOW(),
                    index_status = EXCLUDED.index_status,
                    repository_stats = EXCLUDED.repository_stats,
                    updated_at = NOW()
            """, (
                repo.repo_id,
                repo.repo_url,
                repo.name,
                repo.owner,
                f"Repository: {repo.repo_id}",
                'indexing',
                repo_info
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update repository record: {e}")
    
    async def _process_file_batch(self, repo, files: List[Dict[str, Any]], base_url: str) -> List[Dict[str, Any]]:
        """Process a batch of files."""
        chunks = []
        
        for file_info in files:
            try:
                # Read file content
                content = repo.read_file_content(file_info['path'])
                if not content or len(content.strip()) == 0:
                    continue
                
                # Skip files that are too large
                if len(content) > self.max_file_size:
                    logger.warning(f"Skipping large file: {file_info['path']} ({len(content)} bytes)")
                    continue
                
                # Create file chunks
                file_chunks = await self._chunk_file_content(
                    content, file_info, base_url, repo.repo_id
                )
                chunks.extend(file_chunks)
                
            except Exception as e:
                logger.warning(f"Error processing file {file_info['path']}: {e}")
                continue
        
        return chunks
    
    async def _chunk_file_content(
        self, 
        content: str, 
        file_info: Dict[str, Any], 
        base_url: str, 
        repo_id: str
    ) -> List[Dict[str, Any]]:
        """Chunk file content for embedding."""
        chunks = []
        
        # Create file URL
        file_url = f"{base_url}/blob/main/{file_info['path']}"
        
        # Basic chunking strategy - split by functions/classes for code files
        if file_info['language'] in ['python', 'javascript', 'typescript', 'java', 'cpp']:
            chunks.extend(await self._chunk_code_file(content, file_info, file_url, repo_id))
        else:
            # Generic text chunking for other files
            chunks.extend(await self._chunk_text_file(content, file_info, file_url, repo_id))
        
        return chunks
    
    async def _chunk_code_file(
        self, 
        content: str, 
        file_info: Dict[str, Any], 
        file_url: str, 
        repo_id: str
    ) -> List[Dict[str, Any]]:
        """Chunk code files by functions/classes."""
        chunks = []
        lines = content.split('\n')
        
        # Simple heuristic-based chunking
        # In production, you'd use proper AST parsing with tree-sitter
        
        current_chunk = []
        current_chunk_lines = []
        chunk_number = 0
        in_function = False
        indent_level = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())
            
            # Detect function/class definitions (simple heuristic)
            is_definition = (
                (file_info['language'] == 'python' and 
                 (stripped.startswith('def ') or stripped.startswith('class '))) or
                (file_info['language'] in ['javascript', 'typescript'] and 
                 ('function ' in stripped or '=>' in stripped)) or
                (file_info['language'] == 'java' and 
                 any(keyword in stripped for keyword in ['public ', 'private ', 'protected ']) and 
                 ('(' in stripped and ')' in stripped))
            )
            
            if is_definition and not in_function:
                # Start new chunk if we have content
                if current_chunk and len('\n'.join(current_chunk)) > 100:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk), 
                        file_info, 
                        file_url, 
                        repo_id,
                        chunk_number,
                        current_chunk_lines,
                        content
                    ))
                    chunk_number += 1
                
                current_chunk = [line]
                current_chunk_lines = [i]
                in_function = True
                indent_level = current_indent
                
            elif in_function and current_indent <= indent_level and stripped and not stripped.startswith('#'):
                # End of function
                current_chunk.append(line)
                current_chunk_lines.append(i)
                
                if len('\n'.join(current_chunk)) > 100:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk), 
                        file_info, 
                        file_url, 
                        repo_id,
                        chunk_number,
                        current_chunk_lines,
                        content
                    ))
                    chunk_number += 1
                
                current_chunk = []
                current_chunk_lines = []
                in_function = False
                
            else:
                current_chunk.append(line)
                current_chunk_lines.append(i)
                
                # Prevent chunks from getting too large
                if len('\n'.join(current_chunk)) > 2000:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk), 
                        file_info, 
                        file_url, 
                        repo_id,
                        chunk_number,
                        current_chunk_lines,
                        content
                    ))
                    chunk_number += 1
                    current_chunk = []
                    current_chunk_lines = []
                    in_function = False
        
        # Add remaining content
        if current_chunk and len('\n'.join(current_chunk)) > 50:
            chunks.append(self._create_chunk(
                '\n'.join(current_chunk), 
                file_info, 
                file_url, 
                repo_id,
                chunk_number,
                current_chunk_lines,
                content
            ))
        
        return chunks
    
    async def _chunk_text_file(
        self, 
        content: str, 
        file_info: Dict[str, Any], 
        file_url: str, 
        repo_id: str
    ) -> List[Dict[str, Any]]:
        """Chunk text files by paragraphs/sections."""
        chunks = []
        
        # For markdown files, split by headers
        if file_info['extension'] == '.md':
            sections = content.split('\n#')
            chunk_number = 0
            
            for i, section in enumerate(sections):
                if i > 0:
                    section = '#' + section  # Restore header
                
                if len(section.strip()) > 100:
                    chunks.append(self._create_chunk(
                        section.strip(), 
                        file_info, 
                        file_url, 
                        repo_id,
                        chunk_number,
                        [],  # Line numbers not tracked for text
                        content
                    ))
                    chunk_number += 1
        
        else:
            # Generic chunking for other text files
            chunk_size = 1000
            overlap = 200
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) > 100:
                    chunks.append(self._create_chunk(
                        chunk_text.strip(), 
                        file_info, 
                        file_url, 
                        repo_id,
                        i // (chunk_size - overlap),
                        [],
                        content
                    ))
        
        return chunks
    
    def _create_chunk(
        self, 
        chunk_content: str, 
        file_info: Dict[str, Any], 
        file_url: str, 
        repo_id: str,
        chunk_number: int,
        line_numbers: List[int],
        full_content: str
    ) -> Dict[str, Any]:
        """Create a chunk dictionary."""
        return {
            'url': file_url,
            'chunk_number': chunk_number,
            'content': chunk_content,
            'full_content': full_content,
            'metadata': {
                'file_path': file_info['path'],
                'file_name': file_info['name'],
                'language': file_info['language'],
                'file_extension': file_info['extension'],
                'file_size': file_info['size'],
                'chunk_size': len(chunk_content),
                'line_count': len(chunk_content.split('\n')),
                'start_line': min(line_numbers) if line_numbers else 0,
                'end_line': max(line_numbers) if line_numbers else 0,
                'repo_id': repo_id,
                'processor': self.name,
                'content_type': 'code' if file_info['language'] != 'unknown' else 'documentation'
            }
        }
    
    async def _generate_repository_summary(self, repo_info: Dict[str, Any], sample_chunks: List[Dict[str, Any]]) -> str:
        """Generate a summary for the repository."""
        try:
            # Use repository info and sample content to create summary
            repo_name = repo_info.get('name', 'Unknown')
            owner = repo_info.get('owner', 'Unknown')
            
            # Analyze file types in sample
            languages = set()
            for chunk in sample_chunks:
                lang = chunk['metadata'].get('language', 'unknown')
                if lang != 'unknown':
                    languages.add(lang)
            
            language_str = ', '.join(sorted(languages)) if languages else 'various languages'
            
            summary = f"GitHub repository {owner}/{repo_name} containing code in {language_str}."
            
            if repo_info.get('commit_message'):
                summary += f" Latest commit: {repo_info['commit_message'][:100]}..."
            
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate repository summary: {e}")
            return f"GitHub repository {repo_info.get('repo_id', 'unknown')}"
    
    async def cleanup(self):
        """Clean up processor resources."""
        try:
            await repo_manager.cleanup_all()
        except Exception as e:
            logger.error(f"Error during GitHub processor cleanup: {e}")