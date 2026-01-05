FROM python:3.10-slim
FROM ghcr.io/mlflow/mlflow:v3.1.4

ARG APP_ENV=nonprod
ARG GIT_SHA=unknown
ARG GIT_REF=unknown

ENV APP_ENV=${APP_ENV}
ENV GIT_SHA=${GIT_SHA}
ENV GIT_REF=${GIT_REF}


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Add Postgres driver for backend store + boto3 for S3/MinIO artifacts
RUN pip install --no-cache-dir psycopg2-binary boto3

# Copy application code
COPY src/ /app/src/
COPY api/ /app/api/
COPY scripts/ /app/scripts


EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
