FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
ENV FLASK_APP=server.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120"]
