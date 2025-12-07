FROM python:3.10-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy config file
COPY params.yaml ./params.yaml

# 3. Copy project code
COPY app ./app
COPY src ./src
COPY app ./models

# (optional) copy data if needed
# COPY data ./data

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
