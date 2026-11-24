# Self-contained demo image: generates the synthetic dataset and trains both baselines at build
# time, then serves them. Synthetic data only. To serve your own artifacts instead, mount them
# and point CBP_MODELS_DIR / CBP_FEATURES at the mount.
#
#   docker build -t chargeback-prediction .
#   docker run --rm -p 8000:8000 chargeback-prediction    # then open http://localhost:8000/docs

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, in their own layer, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Demo artifacts, built from a fixed seed so every image serves the same models.
ENV PATH=/app/.venv/bin:$PATH MPLBACKEND=Agg
RUN cbp generate --out data/synthetic --seed 7 \
 && cbp features --data data/synthetic --out data/features.csv \
 && cbp train --features data/features.csv --out artifacts/baseline \
 && rm -rf data/synthetic

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --system --uid 10001 --no-create-home cbp
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/data /app/data
COPY --from=build /app/artifacts /app/artifacts
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CBP_MODELS_DIR=/app/artifacts/baseline \
    CBP_FEATURES=/app/data/features.csv \
    CBP_MODEL=logreg
USER cbp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["cbp", "serve", "--host", "0.0.0.0", "--port", "8000"]
