FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY params.yaml .
ENV PYTHONPATH=/app/src SPRINTML_ARTIFACTS=/data WORKERS=2 TF_CPP_MIN_LOG_LEVEL=3

# retraining worker: adds experiment tracking / registry
FROM base AS worker
ENV GIT_PYTHON_REFRESH=quiet
RUN pip install --no-cache-dir mlflow==3.5.1 "sqlalchemy>=2.0.30,<2.1"

FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["sh", "-c", "export PROMETHEUS_MULTIPROC_DIR=/tmp/prom && rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn sprintml.service:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
