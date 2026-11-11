"""Docker, nginx, docker-compose - зона Коля (этапы 3+, 12).

Сервисы compose: db (postgres:16), redis, qdrant, ollama (+init), app, worker, beat, nginx.
Healthchecks, depends_on service_healthy, multi-stage Dockerfile, security headers.
"""
