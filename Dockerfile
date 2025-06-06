# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    wget \
    git \
    git-lfs \
    openssh-client \
    chromium \
    chromium-driver \
    tree \
    file \
    && rm -rf /var/lib/apt/lists/*

# Configure Git for container use
RUN git config --global user.name "SNARF Bot" && \
    git config --global user.email "snarf@localhost" && \
    git config --global init.defaultBranch main && \
    git config --global safe.directory '*'

# Set environment variables for Chromium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_DRIVER=/usr/bin/chromedriver
ENV DISPLAY=:99

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set permissions
RUN chmod +x /app/entrypoint.sh 2>/dev/null || echo "No entrypoint.sh found"

# Expose port
EXPOSE 8000

# Set default command
CMD ["python", "main.py"]