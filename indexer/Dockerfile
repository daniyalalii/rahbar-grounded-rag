FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

WORKDIR /app

# Upgrade first so Debian security fixes (zlib1g) are included, then install the OCR tools.
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -u 10001 -d /tmp -s /usr/sbin/nologin appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./pipeline ./pipeline

USER 10001

CMD ["python", "-m", "pipeline.run"]