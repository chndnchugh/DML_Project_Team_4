Docker Setup for Kidney MLOps Project
======================================

Files in this ZIP:

1. Dockerfile
   - Builds an image for the FastAPI app defined in app/main.py
   - Installs dependencies from requirements.txt
   - Exposes port 8000

2. docker-compose.yml
   - Service: api
   - Builds from current directory
   - Maps host port 8000 -> container 8000

3. .dockerignore
   - Excludes venv, data, mlruns, etc. from the build context.

Basic commands:

Build image:
  docker compose build

Run container:
  docker compose up

Stop container:
  docker compose down

Then open:
  http://127.0.0.1:8000/docs
