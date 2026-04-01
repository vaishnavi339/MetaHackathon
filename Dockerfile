FROM python:3.10-slim

WORKDIR /app

COPY env ./env
COPY server ./server
COPY inference.py ./inference.py
COPY openenv.yaml ./openenv.yaml
COPY README.md ./README.md
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
