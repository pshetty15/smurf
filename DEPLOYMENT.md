# SMURF Deployment Guide

## 🚀 Complete Deployment Steps

### 1. Rename Project to SMURF

```bash
# Run the renaming script
./rename-to-smurf.sh

# Clean up old containers
docker-compose down -v

# Rename the directory
cd ..
mv snarf snurf
cd snurf
```

### 2. Pre-Deployment Setup

#### 2.1 Environment Configuration
```bash
# Create production .env file
cp .env.example .env

# Edit .env with production values:
# - OPENAI_API_KEY=your-production-key
# - MODEL_CHOICE=gpt-4o-mini
# - LOG_LEVEL=WARNING
# - USE_CONTEXTUAL_EMBEDDINGS=true
```

#### 2.2 Production Docker Compose
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  postgres:
    restart: always
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secure-password-here}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups

  snurf-mcp:
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  snurf-api:
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
```

### 3. Deployment Options

## Option A: Single Server Deployment

### Prerequisites
- Ubuntu 20.04+ or similar Linux server
- Docker and Docker Compose installed
- 4GB+ RAM recommended
- 20GB+ storage

### Steps
```bash
# 1. Clone repository to server
git clone https://github.com/your-repo/snurf.git
cd snurf

# 2. Set up environment
cp .env.example .env
# Edit .env with production values

# 3. Build and start services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 4. Verify deployment
docker-compose ps
curl http://localhost:8080/health
```

## Option B: Cloud Deployment (AWS/GCP/Azure)

### AWS EC2 Deployment
```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. Install Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER

# 3. Clone and deploy
git clone https://github.com/your-repo/snurf.git
cd snurf
./deploy-aws.sh
```

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml -c docker-compose.prod.yml snurf
```

## Option C: Kubernetes Deployment

Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: snurf-mcp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: snurf-mcp
  template:
    metadata:
      labels:
        app: snurf-mcp
    spec:
      containers:
      - name: snurf-mcp
        image: snurf:latest
        command: ["python", "/app/src/mcp_server_standalone.py"]
        env:
        - name: POSTGRES_HOST
          value: postgres-service
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: snurf-secrets
              key: openai-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: snurf-mcp-service
spec:
  selector:
    app: snurf-mcp
  ports:
  - port: 8000
    targetPort: 8000
```

Deploy to Kubernetes:
```bash
# Create secrets
kubectl create secret generic snurf-secrets --from-literal=openai-api-key=$OPENAI_API_KEY

# Deploy
kubectl apply -f k8s-deployment.yaml
```

## Option D: Serverless Deployment (AWS Lambda)

### Container Image for Lambda
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Lambda handler
CMD ["lambda_handler.handler"]
```

### 4. Post-Deployment Configuration

#### 4.1 SSL/TLS Setup (with Nginx)
```nginx
server {
    listen 443 ssl;
    server_name snurf.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/snurf.crt;
    ssl_certificate_key /etc/ssl/private/snurf.key;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 4.2 Monitoring Setup
```bash
# Install monitoring stack
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Includes:
# - Prometheus (metrics)
# - Grafana (dashboards)
# - Loki (logs)
```

### 5. Production Checklist

- [ ] Environment variables configured
- [ ] Secure passwords set
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring enabled
- [ ] Health checks configured
- [ ] Auto-restart enabled
- [ ] Resource limits set
- [ ] Logging configured

### 6. Backup Strategy

```bash
# Database backup script
cat > backup-snurf.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
docker-compose exec -T postgres pg_dump -U snurf_user snurf_db > backups/snurf-$DATE.sql
# Keep last 7 days
find backups -name "snurf-*.sql" -mtime +7 -delete
EOF

# Add to crontab
crontab -e
# Add: 0 2 * * * /path/to/backup-snurf.sh
```

### 7. Scaling Considerations

#### Horizontal Scaling
- Use multiple MCP server instances behind a load balancer
- PostgreSQL read replicas for search queries
- Redis cluster for distributed caching

#### Performance Optimization
- Enable pgbouncer for connection pooling
- Use CDN for static assets
- Implement rate limiting
- Cache embeddings in Redis

### 8. Maintenance

#### Update Process
```bash
# 1. Pull latest changes
git pull

# 2. Build new images
docker-compose build

# 3. Rolling update
docker-compose up -d --no-deps --scale snurf-mcp=2
docker-compose up -d --no-deps snurf-mcp
```

#### Health Monitoring
```bash
# Check service health
curl http://localhost:8080/health

# View logs
docker-compose logs -f snurf-mcp

# Database status
docker-compose exec postgres pg_isready
```

### 9. Security Hardening

- Use secrets management (Vault, AWS Secrets Manager)
- Enable database SSL
- Implement API rate limiting
- Use non-root user in containers
- Regular security updates
- Network isolation with Docker networks

### 10. Troubleshooting

#### Common Issues
1. **MCP not responding**: Check logs with `docker-compose logs snurf-mcp`
2. **Database connection failed**: Verify credentials and network
3. **Out of memory**: Increase Docker memory limits
4. **Slow performance**: Check embedding generation and database indexes

#### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG docker-compose up snurf-mcp
```

## 🚢 Quick Deploy Script

Save as `deploy-snurf.sh`:
```bash
#!/bin/bash
set -e

echo "🚀 Deploying SNURF..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required"; exit 1; }

# Setup
[ ! -f .env ] && cp .env.example .env && echo "⚠️  Edit .env file!"

# Deploy
docker-compose pull
docker-compose up -d

# Wait for services
sleep 10

# Verify
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "✅ SNURF deployed successfully!"
    echo "🌐 API: http://localhost:8080"
    echo "🔌 MCP: docker compose exec -T snurf-mcp python /app/src/mcp_server_standalone.py"
else
    echo "❌ Deployment failed. Check logs: docker-compose logs"
fi
```

## Ready to deploy! 🎉