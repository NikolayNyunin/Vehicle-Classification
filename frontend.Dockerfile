FROM python:3.12.6-slim-bullseye AS builder

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN python -m pip install --no-cache-dir poetry==1.8.2 \
    && poetry config virtualenvs.in-project true \
    && poetry install --only main,frontend --no-dev --no-interaction --no-ansi

FROM python:3.12.6-slim-bullseye

WORKDIR /app

COPY --from=builder /app /app
COPY frontend /app/frontend

EXPOSE 8501

CMD ["/app/.venv/bin/streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
