FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps required for building some Python packages and for NLTK downloads
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure setup script is executable and run it to fetch NLTK data used by the app
RUN chmod +x /app/setup.sh || true
RUN /app/setup.sh || true

# Expose Streamlit default port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
