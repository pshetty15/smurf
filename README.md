# smurf - semantic multi-source unified retrieval framework

A containerized RAG (Retrieval-Augmented Generation) system that intelligently processes and indexes content from multiple sources. smurf provides semantic search capabilities across web pages, GitHub repositories, and other sources using vector embeddings.

## 🚀 Quick Start - Local Docker Setup

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux) with Docker Compose
- Git
- OpenAI API key (get one from https://platform.openai.com)
- Minimum 4GB RAM available for Docker

### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/your-repo/smurf.git
cd smurf

# Verify Docker is running
docker --version
docker-compose --version
```

### 2. Configure Environment
```bash
# Copy the environment template
cp .env.example .env

# Edit .env file and add your OpenAI API key
# On Windows: notepad .env
# On Mac/Linux: nano .env or vim .env

# Required: Set your OpenAI API key
OPENAI_API_KEY=sk-...your-key-here...

# Optional: Adjust other settings as needed
# MODEL_CHOICE=gpt-4o-mini (default, or use gpt-3.5-turbo for lower cost)
# USE_CONTEXTUAL_EMBEDDINGS=true (enhances search quality)
```

### 3. Start smurf Services
```bash
# Build and start all services in detached mode
docker-compose up -d --build

# This will start:
# - PostgreSQL with pgvector extension (port 5432)
# - Redis cache (port 6379)
# - smurf API server (port 8080)
# - smurf application container

# Check if services are running
docker-compose ps

# View logs (useful for troubleshooting)
docker-compose logs -f

# View specific service logs
docker-compose logs -f smurf-api
```

### 4. Verify Installation
```bash
# Check API health
curl http://localhost:8080/health

# Expected response: {"status": "healthy", "database": "connected"}
```

### 5. Access smurf
- **REST API**: http://localhost:8080/docs
- **Interactive CLI**: `docker-compose exec smurf-app python main.py`
- **PostgreSQL Database**: 
  - Host: localhost
  - Port: 5432
  - Database: smurf_db
  - User: smurf_user
  - Password: smurf_password

## 🏗️ Architecture

### Services
- **smurf-postgres**: PostgreSQL with pgvector extension
- **smurf-redis**: Redis for caching and session management
- **smurf-app**: Main application (interactive mode)
- **smurf-api**: REST API server  

### Components
- **Database Layer**: Unified PostgreSQL interface with vector embeddings
- **Processor System**: Modular content processors (Web, GitHub, etc.)
- **Router**: Intelligent routing to appropriate processors
- **API Server**: FastAPI-based REST interface

## 📡 API Usage

### Crawl a URL
```bash
# Web documentation
curl -X POST "http://localhost:8080/crawl" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/tutorial/"}'

# GitHub repository
curl -X POST "http://localhost:8080/crawl" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/fastapi/fastapi"}'
```

### Search the knowledge base
```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "python functions", "limit": 5}'
```

### Get available sources
```bash
curl "http://localhost:8080/sources"
```

### Batch crawl multiple URLs
```bash
curl -X POST "http://localhost:8080/batch-crawl" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://docs.python.org/3/tutorial/",
      "https://github.com/fastapi/fastapi",
      "https://github.com/openai/openai-python"
    ],
    "smart_crawl": true
  }'
```

## 📝 Interactive Mode

### Interactive Mode Examples

```bash
# Start interactive session
docker-compose exec smurf-app python main.py

# Example commands:
smurf> smurf https://docs.python.org/3/tutorial/
smurf> smurf https://github.com/fastapi/fastapi
smurf> search "python functions"
smurf> search "FastAPI dependency injection"
smurf> sources
smurf> quit
```

### API Examples

```bash
# Index a URL
curl -X POST "http://localhost:8080/process" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://docs.python.org/3/tutorial/"}'

# Search indexed content
curl -X POST "http://localhost:8080/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "python functions", "limit": 5}'

# List all sources
curl "http://localhost:8080/sources"

# Health check
curl "http://localhost:8080/health"
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
MODEL_CHOICE=gpt-4o-mini
USE_CONTEXTUAL_EMBEDDINGS=true
USE_AGENTIC_RAG=false
LOG_LEVEL=INFO
MAX_CONCURRENT_CRAWLS=5
MAX_FILE_SIZE=1048576

# AWS Bedrock Configuration
AWS_PROFILE=your-profile
AWS_REGION=us-west-2
BEDROCK_ENDPOINT_URL=https://bedrock-runtime.us-west-2.amazonaws.com

# Database Configuration  
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=smurf_db
POSTGRES_USER=smurf_user
POSTGRES_PASSWORD=smurf_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080

# Application Mode
SMURF_MODE=interactive  # or "demo"

# Embedding Configuration
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v1
BEDROCK_TEXT_MODEL=anthropic.claude-3-haiku-20240307-v1:0
```

### Feature Flags
- **USE_CONTEXTUAL_EMBEDDINGS**: Enhance chunks with LLM-generated context
- **USE_AGENTIC_RAG**: Extract and index code examples
- **MAX_CONCURRENT_CRAWLS**: Limit parallel processing

## 📊 Database Schema

### Core Tables
- `sources`: Source metadata and summaries
- `crawled_pages`: Document chunks with embeddings
- `code_examples`: Code snippets with summaries

### GitHub Extensions
- `repositories`: Repository tracking
- `code_structures`: Detailed code analysis
- `repository_relationships`: Dependency mapping

## 🔍 Search Capabilities

### Vector Search
```python
# Semantic similarity search
results = db.search_documents("machine learning concepts", match_count=10)
```

### Keyword Search
```python
# Text-based search
results = db.keyword_search_documents("async def", match_count=5)
```

### Filtered Search
```python
# Search within specific sources
results = db.search_documents("API design", source_filter="docs.python.org")
```

## 🐳 Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Access database
docker-compose exec postgres psql -U smurf_user -d smurf_db

# Interactive shell
docker-compose exec smurf-app bash
```

## 🔧 Development

### Local Development Workflow
```bash
# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build -d

# Access container shell for debugging
docker-compose exec smurf-app bash

# Run tests (if available)
docker-compose exec smurf-app pytest

# Clean up everything (including volumes)
docker-compose down -v
```

### Adding New Processors
1. Create processor class inheriting from `BaseProcessor`
2. Implement `can_handle()` and `process()` methods
3. Register with `@register_processor` decorator
4. Add to router in initialization

### Database Migrations
Add SQL files to `init-db/` directory (they run in alphabetical order on container start).

## 🚨 Troubleshooting

### Common Issues

#### 1. Docker Desktop Not Running
```bash
# Windows: Start Docker Desktop from Start Menu
# Mac: Start Docker from Applications
# Linux: sudo systemctl start docker
```

#### 2. Port Already in Use
```bash
# Check what's using the ports
# Windows: netstat -ano | findstr :8080
# Mac/Linux: lsof -i :8080

# Change ports in docker-compose.yml if needed
```

#### 3. Database Connection Issues
```bash
# Check database status
docker-compose exec postgres pg_isready -U smurf_user

# View database logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

#### 4. API Not Responding
```bash
# Check API logs
docker-compose logs smurf-api

# Check if container is running
docker-compose ps smurf-api

# Restart API service
docker-compose restart smurf-api
```

#### 5. OpenAI API Errors
```bash
# Verify your API key is set correctly
docker-compose exec smurf-api env | grep OPENAI

# Check for rate limits or quota issues in logs
docker-compose logs smurf-api | grep -i "openai\|error"
```

#### 6. Memory Issues
```bash
# Check resource usage
docker stats

# Reduce concurrent crawls in .env
MAX_CONCURRENT_CRAWLS=3

# Increase Docker memory (Docker Desktop settings)
```

### Debugging Tips
```bash
# Enter container for debugging
docker-compose exec smurf-app bash

# Check Python dependencies
pip list

# Test database connection manually
python -c "from src.core.database import Database; db = Database(); print('Connected!')"

# Check environment variables
env | sort
```

## 📈 Monitoring

### Health Checks
- **API Health**: http://localhost:8080/health
- **System Stats**: http://localhost:8080/stats

### Logs
```bash
# All logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f smurf-api
docker-compose logs -f postgres
docker-compose logs -f redis

# Save logs to file
docker-compose logs > smurf-logs.txt
```

## 🛡️ Security Notes

- Change default passwords in production
- Configure CORS properly for API
- Use environment variables for secrets
- Enable database SSL in production
- Don't commit .env file to version control

## 📚 Additional Resources

- [MCP Theory and Architecture](./MCP_THEORY.md)
- [API Documentation](./API_DOCS.md)
- [Deployment Guide](./DEPLOYMENT.md)

## 📝 License

[Add your license here]