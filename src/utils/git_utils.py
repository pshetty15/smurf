"""
Git utilities for smurf GitHub processor.
Handles repository cloning, file analysis, and code structure extraction.
"""

import os
import shutil
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from urllib.parse import urlparse
import asyncio
import subprocess

import git
from git import Repo, InvalidGitRepositoryError


logger = logging.getLogger(__name__)


class GitRepository:
    """Manages Git repository operations for smurf."""
    
    def __init__(self, repo_url: str, cache_dir: str = None):
        """
        Initialize Git repository handler.
        
        Args:
            repo_url: GitHub repository URL
            cache_dir: Directory to cache cloned repositories
        """
        self.repo_url = repo_url
        self.cache_dir = cache_dir or "/app/data/repos"
        self.repo_path = None
        self.repo = None
        
        # Parse repository info
        parsed = urlparse(repo_url)
        if 'github.com' in parsed.netloc:
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                self.owner = path_parts[0]
                self.name = path_parts[1].replace('.git', '')
                self.repo_id = f"{self.owner}/{self.name}"
            else:
                raise ValueError(f"Invalid GitHub URL: {repo_url}")
        else:
            raise ValueError(f"Only GitHub URLs are supported: {repo_url}")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    async def clone_or_update(self) -> str:
        """
        Clone repository or update if it already exists.
        
        Returns:
            Path to the local repository
        """
        self.repo_path = os.path.join(self.cache_dir, f"{self.owner}_{self.name}")
        
        try:
            if os.path.exists(self.repo_path):
                logger.info(f"Repository exists, updating: {self.repo_path}")
                await self._update_repository()
            else:
                logger.info(f"Cloning repository: {self.repo_url}")
                await self._clone_repository()
            
            return self.repo_path
            
        except Exception as e:
            logger.error(f"Failed to clone/update repository {self.repo_url}: {e}")
            # Clean up on failure
            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path, ignore_errors=True)
            raise
    
    async def _clone_repository(self):
        """Clone the repository."""
        try:
            # Use subprocess for better control over the clone process
            cmd = [
                'git', 'clone', 
                '--depth', '1',  # Shallow clone for faster cloning
                '--single-branch',
                self.repo_url,
                self.repo_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Git clone failed: {stderr.decode()}")
            
            self.repo = Repo(self.repo_path)
            logger.info(f"Successfully cloned repository to {self.repo_path}")
            
        except Exception as e:
            logger.error(f"Clone operation failed: {e}")
            raise
    
    async def _update_repository(self):
        """Update existing repository."""
        try:
            self.repo = Repo(self.repo_path)
            
            # Fetch latest changes
            origin = self.repo.remotes.origin
            origin.fetch()
            
            # Reset to latest
            self.repo.heads.main.reset(origin.refs.main, index=True, working_tree=True)
            
            logger.info(f"Successfully updated repository: {self.repo_path}")
            
        except Exception as e:
            logger.warning(f"Update failed, will re-clone: {e}")
            # If update fails, remove and re-clone
            shutil.rmtree(self.repo_path, ignore_errors=True)
            await self._clone_repository()
    
    def get_file_list(self, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of files in the repository.
        
        Args:
            extensions: File extensions to filter (e.g., ['.py', '.js'])
            
        Returns:
            List of file information dictionaries
        """
        if not self.repo_path or not os.path.exists(self.repo_path):
            return []
        
        files = []
        extensions = extensions or ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php']
        
        for root, dirs, filenames in os.walk(self.repo_path):
            # Skip hidden directories and common build/cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                'node_modules', '__pycache__', 'build', 'dist', 'target', 'vendor'
            }]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, self.repo_path)
                
                # Skip hidden files and large files
                if filename.startswith('.') or os.path.getsize(file_path) > 1024 * 1024:  # 1MB limit
                    continue
                
                # Filter by extension
                file_ext = os.path.splitext(filename)[1].lower()
                if extensions and file_ext not in extensions:
                    continue
                
                try:
                    # Get file stats
                    stat = os.stat(file_path)
                    
                    files.append({
                        'path': relative_path,
                        'full_path': file_path,
                        'name': filename,
                        'extension': file_ext,
                        'size': stat.st_size,
                        'language': self._detect_language(file_ext),
                        'modified_time': stat.st_mtime
                    })
                    
                except OSError:
                    continue
        
        return files
    
    def read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read content of a file in the repository.
        
        Args:
            file_path: Relative path to the file
            
        Returns:
            File content or None if not readable
        """
        if not self.repo_path:
            return None
        
        full_path = os.path.join(self.repo_path, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None
    
    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get repository metadata.
        
        Returns:
            Dictionary with repository information
        """
        if not self.repo:
            return {}
        
        try:
            # Get basic info
            info = {
                'url': self.repo_url,
                'owner': self.owner,
                'name': self.name,
                'repo_id': self.repo_id,
                'local_path': self.repo_path,
            }
            
            # Get commit info
            try:
                latest_commit = self.repo.head.commit
                info.update({
                    'latest_commit': latest_commit.hexsha,
                    'commit_date': latest_commit.committed_datetime.isoformat(),
                    'commit_message': latest_commit.message.strip(),
                    'author': str(latest_commit.author)
                })
            except Exception:
                pass
            
            # Get branch info
            try:
                info['branch'] = self.repo.active_branch.name
            except Exception:
                info['branch'] = 'main'
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return {'error': str(e)}
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.rst': 'rst',
            '.txt': 'text'
        }
        
        return language_map.get(extension.lower(), 'unknown')
    
    def cleanup(self):
        """Clean up cloned repository."""
        if self.repo_path and os.path.exists(self.repo_path):
            try:
                shutil.rmtree(self.repo_path)
                logger.info(f"Cleaned up repository: {self.repo_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup repository {self.repo_path}: {e}")


class GitRepositoryManager:
    """Manages multiple Git repositories with caching."""
    
    def __init__(self, cache_dir: str = "/app/data/repos", max_repos: int = 50):
        """
        Initialize repository manager.
        
        Args:
            cache_dir: Directory to cache repositories
            max_repos: Maximum number of repositories to keep cached
        """
        self.cache_dir = cache_dir
        self.max_repos = max_repos
        self.repositories: Dict[str, GitRepository] = {}
        
        os.makedirs(cache_dir, exist_ok=True)
    
    async def get_repository(self, repo_url: str) -> GitRepository:
        """
        Get or create a repository instance.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            GitRepository instance
        """
        # Create repository ID from URL
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip('/').split('/')
        repo_id = f"{path_parts[0]}/{path_parts[1].replace('.git', '')}"
        
        if repo_id not in self.repositories:
            # Clean up old repositories if we've hit the limit
            if len(self.repositories) >= self.max_repos:
                await self._cleanup_oldest_repository()
            
            self.repositories[repo_id] = GitRepository(repo_url, self.cache_dir)
        
        return self.repositories[repo_id]
    
    async def _cleanup_oldest_repository(self):
        """Remove the oldest cached repository."""
        if not self.repositories:
            return
        
        # For simplicity, remove the first repository
        # In a more sophisticated implementation, you'd track access times
        oldest_id = next(iter(self.repositories))
        oldest_repo = self.repositories.pop(oldest_id)
        oldest_repo.cleanup()
        
        logger.info(f"Cleaned up old repository: {oldest_id}")
    
    async def cleanup_all(self):
        """Clean up all cached repositories."""
        for repo in self.repositories.values():
            repo.cleanup()
        self.repositories.clear()
        logger.info("Cleaned up all cached repositories")


# Global repository manager instance
repo_manager = GitRepositoryManager()