FROM python:3.12.6-slim-bullseye AS builder

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN python -m pip install --no-cache-dir poetry==1.8.2 \
    && poetry config virtualenvs.in-project true \
    && poetry install --only main,backend --no-dev --no-interaction --no-ansi

FROM python:3.12.6-slim-bullseye

WORKDIR /app

ENV PYTHONPATH=/app

COPY --from=builder /app /app
COPY api /app/api
COPY models /app/models

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
