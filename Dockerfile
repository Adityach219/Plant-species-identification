FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY webapp/ ./webapp/
COPY config/ ./config/
COPY models/ ./models/

# Create directories for data and logs
RUN mkdir -p data logs

# Expose ports for web applications
EXPOSE 8501 5000

# Default command (can be overridden)
CMD ["python", "-m", "streamlit", "run", "webapp/app.py", "--server.port=8501", "--server.address=0.0.0.0"]