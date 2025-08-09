# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for certain Python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/processed/scaler.pkl ./data/processed/
COPY data/processed/metadata.json ./data/processed/
COPY models/artifacts/ ./models/artifacts/

# Create non-root user for security
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]