FROM python:3.10-slim

WORKDIR /app

# Install system deps for lxml and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

COPY app/ app/
COPY ml/models/ ml/models/

EXPOSE 18500

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18500"]
