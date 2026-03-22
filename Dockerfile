FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"
ENV TRANSFORMERS_OFFLINE=1

COPY . .

RUN mkdir -p docs faiss_index

EXPOSE 8000

# Use shell form so $PORT variable is expanded by Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}