# syntax=docker/dockerfile:1
# Images come from Red Hat's registry (log in first: `docker login registry.redhat.io`).

# ---- 1. build the React UI ----
FROM registry.redhat.io/ubi9/nodejs-22-minimal:latest AS ui
USER 0
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. python dependencies + embedding model baked in (so the runtime FS can be read-only) ----
FROM registry.redhat.io/ubi9/python-312-minimal:latest AS deps
USER 0
WORKDIR /build
COPY backend/requirements.txt ./
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN /opt/venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# ---- 3. runtime ----
FROM registry.redhat.io/ubi9/python-312-minimal:latest
USER 0
RUN useradd -m -u 1001 appuser
COPY --from=deps /opt/venv /opt/venv
COPY --from=deps --chown=1001:0 /opt/models /opt/models
WORKDIR /app
COPY --chown=1001:0 backend/app ./app
COPY --chown=1001:0 samples ./samples
COPY --from=ui --chown=1001:0 /ui/dist ./static
ENV PATH=/opt/venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=container \
    SAMPLES_DIR=/app/samples \
    STATIC_DIR=/app/static
USER 1001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import socket,urllib.request as u; u.urlopen('http://%s:8000/healthz' % socket.gethostbyname(socket.gethostname()), timeout=2)"
CMD ["python", "-m", "app.main"]
