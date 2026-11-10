FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY params.yaml .
ENV PYTHONPATH=/app/src SPRINTML_ARTIFACTS=/data WORKERS=2 TF_CPP_MIN_LOG_LEVEL=3
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["sh", "-c", "uvicorn sprintml.service:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
