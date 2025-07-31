# Ultra-Fast Context-AI Dockerfile
# For deployment on Koyeb, Render, or any cloud platform

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install -r requirements.txt
RUN python3 nltk_setup.py

# Copy application code
COPY app/ ./app/
COPY README.md ./

# Expose port
EXPOSE 8000

# Default command to run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
