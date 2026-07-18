FROM python:3.12-slim AS base

WORKDIR /app

# System deps kept minimal; scipy wheels are prebuilt.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8000

# API layer not yet implemented; this will be the entrypoint once
# risk_service.api.main:app is added.
CMD ["uvicorn", "risk_service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
