"""
Database layer for smurf - extracted and refactored from utils.py
Provides unified interface for all source processors
"""
import os
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
import time
from urllib.parse import urlparse


class Database:
    """
    Unified database interface for all processors.
    Handles PostgreSQL with pgvector for embeddings.
    """
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv("POSTGRES_HOST", "localhost"),
            'port': os.getenv("POSTGRES_PORT", "5432"),
            'database': os.getenv("POSTGRES_DB", "mcp_crawl4ai"),
            'user': os.getenv("POSTGRES_USER", "postgres"),
            'password': os.getenv("POSTGRES_PASSWORD", "password"),
        }
    
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(
            cursor_factory=RealDictCursor,
            **self.connection_params
        )
    
    async def initialize(self):
        """Initialize database connection and verify connectivity."""
        try:
            conn = self.get_connection()
            conn.close()
            print("PostgreSQL connection successful")
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            raise
    
    def store_documents(
        self,
        urls: List[str], 
        chunk_numbers: List[int],
        contents: List[str], 
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        processor_name: str = "web",
        batch_size: int = 20
    ) -> None:
        """
        Store documents in the database with embeddings.
        Unified interface for all processors.
        """
        from .embeddings import create_embeddings_batch, generate_contextual_embedding
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Delete existing records for these URLs
            unique_urls = list(set(urls))
            if unique_urls:
                cursor.execute(
                    "DELETE FROM crawled_pages WHERE url = ANY(%s)",
                    (unique_urls,)
                )
                conn.commit()
            
            # Check if contextual embeddings are enabled
            use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
            
            # Process in batches to avoid memory issues
            for i in range(0, len(contents), batch_size):
                batch_end = min(i + batch_size, len(contents))
                
                # Get batch slices
                batch_urls = urls[i:batch_end]
                batch_chunk_numbers = chunk_numbers[i:batch_end]
                batch_contents = contents[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                
                # Apply contextual embedding if enabled
                if use_contextual_embeddings:
                    # Process in parallel using ThreadPoolExecutor
                    process_args = []
                    for j, content in enumerate(batch_contents):
                        url = batch_urls[j]
                        full_document = url_to_full_document.get(url, "")
                        process_args.append((url, content, full_document))
                    
                    contextual_contents = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        future_to_idx = {executor.submit(self._process_chunk_with_context, arg): idx 
                                        for idx, arg in enumerate(process_args)}
                        
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                result, success = future.result()
                                contextual_contents.append(result)
                                if success:
                                    batch_metadatas[idx]["contextual_embedding"] = True
                            except Exception as e:
                                print(f"Error processing chunk {idx}: {e}")
                                contextual_contents.append(batch_contents[idx])
                    
                    # Sort results back into original order if needed
                    if len(contextual_contents) != len(batch_contents):
                        contextual_contents = batch_contents
                else:
                    contextual_contents = batch_contents
                
                # Create embeddings for the entire batch
                batch_embeddings = create_embeddings_batch(contextual_contents)
                
                # Prepare batch data
                batch_data = []
                for j in range(len(contextual_contents)):
                    chunk_size = len(contextual_contents[j])
                    parsed_url = urlparse(batch_urls[j])
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    # Add processor metadata
                    metadata = {
                        "chunk_size": chunk_size,
                        "processor": processor_name,
                        **batch_metadatas[j]
                    }
                    
                    batch_data.append({
                        "url": batch_urls[j],
                        "chunk_number": batch_chunk_numbers[j],
                        "content": contextual_contents[j],
                        "metadata": Json(metadata),
                        "source_id": source_id,
                        "embedding": batch_embeddings[j]
                    })
                
                # Insert batch with retry logic
                self._insert_batch_with_retry(cursor, conn, batch_data, "crawled_pages")
        
        except Exception as e:
            print(f"Database error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def store_code_examples(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        code_examples: List[str],
        summaries: List[str],
        metadatas: List[Dict[str, Any]],
        processor_name: str = "github",
        batch_size: int = 20
    ):
        """Store code examples with embeddings."""
        from .embeddings import create_embeddings_batch
        
        if not urls:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Delete existing records
            unique_urls = list(set(urls))
            if unique_urls:
                cursor.execute(
                    "DELETE FROM code_examples WHERE url = ANY(%s)",
                    (unique_urls,)
                )
                conn.commit()
            
            # Process in batches
            total_items = len(urls)
            for i in range(0, total_items, batch_size):
                batch_end = min(i + batch_size, total_items)
                batch_texts = []
                
                # Create combined texts for embedding (code + summary)
                for j in range(i, batch_end):
                    combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
                    batch_texts.append(combined_text)
                
                # Create embeddings
                embeddings = create_embeddings_batch(batch_texts)
                
                # Prepare batch data
                batch_data = []
                for j, embedding in enumerate(embeddings):
                    idx = i + j
                    parsed_url = urlparse(urls[idx])
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    metadata = {
                        "processor": processor_name,
                        **metadatas[idx]
                    }
                    
                    batch_data.append({
                        'url': urls[idx],
                        'chunk_number': chunk_numbers[idx],
                        'content': code_examples[idx],
                        'summary': summaries[idx],
                        'metadata': Json(metadata),
                        'source_id': source_id,
                        'embedding': embedding
                    })
                
                # Insert batch
                self._insert_batch_with_retry(cursor, conn, batch_data, "code_examples")
                print(f"Inserted batch {i//batch_size + 1} of {(total_items + batch_size - 1)//batch_size} code examples")
        
        except Exception as e:
            print(f"Database error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def search_documents(
        self,
        query: str, 
        match_count: int = 10, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using vector similarity."""
        from .embeddings import create_embedding
        
        query_embedding = create_embedding(query)
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            filter_json = Json(filter_metadata) if filter_metadata else Json({})
            
            if source_filter:
                cursor.execute(
                    "SELECT * FROM match_crawled_pages(%s, %s, %s, %s)",
                    [query_embedding, match_count, filter_json, source_filter]
                )
            else:
                cursor.execute(
                    "SELECT * FROM match_crawled_pages(%s, %s, %s)",
                    [query_embedding, match_count, filter_json]
                )
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def search_code_examples(
        self,
        query: str, 
        match_count: int = 10, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search code examples using vector similarity."""
        from .embeddings import create_embedding
        
        enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
        query_embedding = create_embedding(enhanced_query)
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            filter_json = Json(filter_metadata) if filter_metadata else Json({})
            
            cursor.execute(
                "SELECT * FROM match_code_examples(%s, %s, %s, %s)",
                [query_embedding, match_count, filter_json, source_id]
            )
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"Error searching code examples: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def keyword_search_documents(
        self,
        query: str, 
        match_count: int = 10, 
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search documents using keyword search (ILIKE)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if source_filter:
                cursor.execute("""
                    SELECT id, url, chunk_number, content, metadata, source_id
                    FROM crawled_pages 
                    WHERE content ILIKE %s AND source_id = %s
                    LIMIT %s
                """, [f'%{query}%', source_filter, match_count])
            else:
                cursor.execute("""
                    SELECT id, url, chunk_number, content, metadata, source_id
                    FROM crawled_pages 
                    WHERE content ILIKE %s
                    LIMIT %s
                """, [f'%{query}%', match_count])
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Get all available sources."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT source_id, summary, total_word_count, processor_name, created_at, updated_at
                FROM sources 
                ORDER BY source_id
            """)
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def update_source_info(
        self, 
        source_id: str, 
        summary: str, 
        word_count: int, 
        processor_name: str = "web"
    ):
        """Update or insert source information."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE sources 
                SET summary = %s, total_word_count = %s, processor_name = %s, updated_at = NOW() 
                WHERE source_id = %s
            """, (summary, word_count, processor_name, source_id))
            
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO sources (source_id, summary, total_word_count, processor_name) 
                    VALUES (%s, %s, %s, %s)
                """, (source_id, summary, word_count, processor_name))
                print(f"Created new source: {source_id}")
            else:
                print(f"Updated source: {source_id}")
            
            conn.commit()
            
        except Exception as e:
            print(f"Error updating source {source_id}: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def _process_chunk_with_context(self, args):
        """Process a single chunk with contextual embedding."""
        from .embeddings import generate_contextual_embedding
        url, content, full_document = args
        return generate_contextual_embedding(full_document, content)
    
    def _insert_batch_with_retry(self, cursor, conn, batch_data, table_name):
        """Insert batch data with retry logic."""
        max_retries = 3
        retry_delay = 1.0
        
        if table_name == "crawled_pages":
            query = """
                INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
                VALUES (%(url)s, %(chunk_number)s, %(content)s, %(metadata)s, %(source_id)s, %(embedding)s)
            """
        elif table_name == "code_examples":
            query = """
                INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
                VALUES (%(url)s, %(chunk_number)s, %(content)s, %(summary)s, %(metadata)s, %(source_id)s, %(embedding)s)
            """
        else:
            raise ValueError(f"Unknown table: {table_name}")
        
        for retry in range(max_retries):
            try:
                cursor.executemany(query, batch_data)
                conn.commit()
                break
            except Exception as e:
                conn.rollback()
                if retry < max_retries - 1:
                    print(f"Error inserting batch (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Try individual inserts as fallback
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            cursor.execute(query, record)
                            conn.commit()
                            successful_inserts += 1
                        except Exception as individual_error:
                            conn.rollback()
                            print(f"Failed to insert individual record: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually")