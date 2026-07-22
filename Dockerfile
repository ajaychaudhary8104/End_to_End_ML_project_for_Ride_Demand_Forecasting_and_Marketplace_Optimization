FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
        curl \
        unzip && \
    rm -rf /var/lib/apt/lists/*

# AWS CLI (needed for Feast + S3 debugging)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
      -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

COPY requirements-prod.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-prod.txt

COPY . .

RUN pip install .

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    mkdir -p /app/feast_data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]