FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Japanese language processing
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (including translation files)
COPY . .

# Create directory for model storage
RUN mkdir -p /app/models

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "api_server.py"]