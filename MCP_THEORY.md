# Model Context Protocol (MCP) in RAG Systems

## Table of Contents
- [What is MCP?](#what-is-mcp)
- [How smurf Implements RAG](#how-smurf-implements-rag)
- [The Theory Behind smurf](#the-theory-behind-smurf)
- [MCP Integration](#mcp-integration)

## What is MCP?

Model Context Protocol (MCP) is a standardized way for AI assistants to interact with external tools and data sources. It allows language models to:

1. **Access Real-time Data**: Query databases, APIs, and file systems
2. **Execute Tools**: Run functions and commands
3. **Maintain Context**: Keep track of conversation state and tool results

### MCP in smurf

smurf includes an MCP server (`mcp_server_standalone.py`) that exposes its RAG capabilities as MCP tools:

- `smurf`: Index content from URLs, GitHub repos, sitemaps, txt files
- `smurf_search`: Semantic search using vector embeddings  
- `smurf_sources`: List all indexed sources

This allows any MCP-compatible AI assistant (like Claude) to leverage smurf's knowledge base.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI technique that combines:

1. **Retrieval**: Finding relevant information from a knowledge base
2. **Augmentation**: Adding that information to the prompt
3. **Generation**: Using an LLM to generate responses with the additional context

### Why RAG Matters

Traditional LLMs have limitations:
- **Knowledge Cutoff**: Training data has a specific cutoff date
- **Hallucination**: May generate plausible but incorrect information
- **No Real-time Data**: Cannot access current information
- **Static Knowledge**: Cannot learn new information without retraining

RAG solves these by providing LLMs with:
- ✅ **Current Information**: Real-time access to updated knowledge
- ✅ **Factual Grounding**: Responses based on retrieved documents
- ✅ **Source Attribution**: Can cite specific sources
- ✅ **Domain Expertise**: Access to specialized knowledge bases

## How smurf Implements RAG

smurf (semantic multi-source unified retrieval framework) is a sophisticated RAG implementation with several advanced features:

### 1. Multi-Source Ingestion
smurf can ingest content from various sources:
- **Web Pages**: Documentation, articles, blogs
- **GitHub Repositories**: Code, README files, documentation
- **Sitemaps**: Bulk processing of website content
- **Plain Text**: Direct content input

### 2. Intelligent Chunking
Content is split into semantic chunks with:
- **Overlapping Windows**: Ensures context preservation
- **Size Optimization**: Balanced for embedding quality
- **Metadata Preservation**: Maintains source information

### 3. Vector Embeddings
Uses AWS Bedrock for creating embeddings:
- **Semantic Understanding**: Captures meaning, not just keywords
- **Multi-dimensional**: Rich representation of content
- **Similarity Search**: Finds conceptually related content

### 4. Contextual Embeddings (Optional)
When enabled, smurf enhances chunks with:
- **Document Context**: Understanding of the full document
- **Relationship Awareness**: How chunks relate to each other
- **Enhanced Retrieval**: Better matching for complex queries

### 5. Hybrid Search
Combines multiple search strategies:
- **Vector Search**: Semantic similarity
- **Keyword Search**: Exact term matching
- **Metadata Filtering**: Source-based filtering

## The Theory Behind smurf

### Vector Embeddings Theory

Embeddings map text to high-dimensional vectors where:
- **Similar concepts** are close together in vector space
- **Dissimilar concepts** are far apart
- **Relationships** are preserved (e.g., king - man + woman ≈ queen)

```
Text: "Python programming tutorial"
↓ Embedding Model
Vector: [0.1, -0.3, 0.8, ..., 0.2] (1536 dimensions)
```

### Chunking Strategy
smurf uses overlapping chunks because:
- **Context Preservation**: Important information isn't split
- **Boundary Issues**: Handles concepts that span chunk boundaries
- **Redundancy**: Multiple chances to retrieve relevant information
- **Quality**: Better embeddings from complete thoughts

### Processor Architecture
smurf's modular processor design allows:
- **Extensibility**: Easy to add new source types
- **Specialization**: Each processor optimized for its source type
- **Maintainability**: Clear separation of concerns
- **Testing**: Each processor can be tested independently

### Search Theory

**Vector Similarity Search:**
```
Query: "How to handle errors in Python?"
↓ Embedding
Query Vector: [0.2, -0.1, 0.9, ..., 0.3]
↓ Cosine Similarity
Most Similar Chunks: Retrieved based on vector distance
```

**Cosine Similarity Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A = query vector
- B = document chunk vector  
- Result range: -1 to 1 (higher = more similar)

### Database Architecture

**PostgreSQL with pgvector:**
- **Vector Storage**: Native vector data type
- **Indexing**: Efficient similarity search with IVFFlat
- **ACID Compliance**: Reliable data storage
- **Scalability**: Handle large knowledge bases

**Schema Design:**
```sql
CREATE TABLE crawled_pages (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER,
    content TEXT,
    metadata JSONB,
    source_id TEXT,
    embedding vector(1536),  -- pgvector type
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON crawled_pages USING ivfflat (embedding vector_cosine_ops);
```

### Advantages of smurf's Design

1. **Up-to-date Information**: Unlike static training data, smurf indexes current content
2. **Source Diversity**: Handles multiple content types and sources
3. **Semantic Search**: Finds conceptually relevant content, not just keyword matches
4. **Scalable Architecture**: Microservices design allows horizontal scaling
5. **API Integration**: REST API and MCP make it easy to integrate
6. **Cost Effective**: AWS Bedrock provides enterprise-grade embeddings

### Limitations and Trade-offs

1. **Embedding Costs**: API calls for each chunk (mitigated by local storage)
2. **Storage Requirements**: Vector embeddings require significant space
3. **Search Latency**: Vector similarity search has computational overhead
4. **Context Windows**: Limited by chunk size and model context limits
5. **Quality Dependence**: Results depend on quality of indexed content

## MCP Integration

### How MCP Enhances RAG

MCP allows smurf to be used as a **live knowledge base** by AI assistants:

1. **Dynamic Indexing**: AI can index new content during conversations
2. **Contextual Search**: Search based on current conversation context
3. **Source Management**: Track and manage indexed sources
4. **Real-time Updates**: Knowledge base stays current

### smurf MCP Tools

```python
# MCP Tool Definitions
tools = [
    {
        "name": "smurf",
        "description": "Index content from URLs into knowledge base",
        "parameters": {"url": "string"}
    },
    {
        "name": "smurf_search", 
        "description": "Search indexed content semantically",
        "parameters": {"query": "string", "limit": "integer"}
    },
    {
        "name": "smurf_sources",
        "description": "List all indexed sources",
        "parameters": {}
    }
]
```

### Example MCP Workflow

1. **Assistant receives question** about a specific technology
2. **Checks if relevant sources** are indexed using `smurf_sources`
3. **If not, indexes new sources** using `smurf` tool
4. **Searches for relevant information** using `smurf_search`
5. **Generates response** using retrieved context

### Benefits of MCP + RAG

- **Conversational Knowledge Building**: Build knowledge base through dialogue
- **Context-Aware Retrieval**: Search based on conversation history
- **Interactive Learning**: AI assistant learns about user's domain
- **Source Transparency**: User knows what sources are being used

## Conclusion

smurf demonstrates how RAG can bridge the gap between static LLM knowledge and dynamic information sources. By combining:

- **Multi-source ingestion** for comprehensive coverage
- **Vector embeddings** for semantic understanding  
- **Intelligent chunking** for context preservation
- **MCP integration** for conversational knowledge building

smurf creates a powerful system that keeps AI assistants grounded in current, factual information while maintaining the flexibility to adapt to new domains and use cases.