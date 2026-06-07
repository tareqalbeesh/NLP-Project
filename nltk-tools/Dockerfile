FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build deps for some scientific wheels (kept minimal — most install from wheels).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Pre-download NLTK corpora into a system-wide location.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} \
        punkt \
        punkt_tab \
        stopwords \
        averaged_perceptron_tagger_eng

# Pre-download spaCy English model.
RUN python -m spacy download en_core_web_sm

# Remove build toolchain to keep the image lean.
RUN apt-get purge -y --auto-remove build-essential

COPY app.py /app/app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
