FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md config.yaml config.production.yaml ./
COPY gym_pentest/ gym_pentest/
COPY agents/ agents/
COPY pentester/ pentester/
COPY evaluation/ evaluation/
COPY utils/ utils/
COPY config_loader.py setup_logging.py custom_sb3_per.py ./

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PENTESTER_CONFIG=/app/config.production.yaml

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD autopentest health --target http://juice-shop:3000 || exit 1

ENTRYPOINT ["autopentest"]
CMD ["scan", "--config", "/app/config.production.yaml", "--target", "http://juice-shop:3000", "--output", "/app/reports/latest"]
