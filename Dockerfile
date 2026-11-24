FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

ARG GIT_SHA=unknown

ENV GIT_SHA=${GIT_SHA} \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app/ app/
COPY ml/drift.py ml/drift.py

USER appuser
EXPOSE 8000

CMD ["sh", "-c", "waitress-serve --host=0.0.0.0 --port=${PORT} --call app:create_app"]
