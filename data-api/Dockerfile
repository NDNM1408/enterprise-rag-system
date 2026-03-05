FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application source
COPY src /app/src

# Expose port (default 3001, configurable via env)
EXPOSE 3001

# Set working directory to src (where main.py is located)
WORKDIR /app/src

# Run the FastAPI application
CMD ["python", "main.py"]
