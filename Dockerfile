FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY env ./env
COPY inference.py ./inference.py
COPY openenv.yaml ./openenv.yaml
COPY README.md ./README.md

RUN pip install --no-cache-dir "openai>=1.30.0" "pydantic>=2.7,<3" "PyYAML>=6.0"

CMD ["python", "inference.py"]
