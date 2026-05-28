# Build context is this directory:
#   docker build -t mlp/license-server:0.1 license_server/
# Run:
#   see license_server/docker-compose.example.yml
#
# Server-only image. The Surrogate Model Training GUI is not in this image — this
# container only serves the floating-seat license API consumed by the GUI client.
# Runtime data (SQLite DB, identity, imported license) lives under /data so it
# persists across container restarts via a named volume or bind mount.

FROM python:3.12-slim AS builder

WORKDIR /tmp/build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim

# Non-root runtime user. /data is the writable runtime root the server uses
# for its SQLite DB, server identity, and the active imported license file.
RUN useradd --uid 1000 --create-home --shell /bin/bash mlp \
 && mkdir -p /data \
 && chown mlp:mlp /data

COPY --from=builder /install /usr/local

# Build context is GUI/license_server/, so this copies the package source to
# /app/license_server/. /app is on PYTHONPATH so `license_server.main:app` resolves.
WORKDIR /app
COPY --chown=mlp:mlp . /app/license_server

USER mlp

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLP_LICENSE_SERVER_RUNTIME_DIR=/data

VOLUME ["/data"]
EXPOSE 27850

CMD ["uvicorn", "license_server.main:app", "--host", "0.0.0.0", "--port", "27850"]
