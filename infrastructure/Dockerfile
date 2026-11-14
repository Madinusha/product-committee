# Multi-stage: builder ставит зависимости через Poetry, runtime забирает готовый venv
FROM python:3.12-slim AS builder

ENV POETRY_VERSION=2.1.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY backend ./backend
COPY scripts ./scripts
COPY run.py pyproject.toml poetry.lock README.md ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["python", "run.py"]
