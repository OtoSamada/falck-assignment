FROM ghcr.io/astral-sh/uv:0.13.0 AS uv

FROM python:3.12-slim-bookworm AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.12-slim-bookworm

ENV HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

RUN mkdir -p /app/data/raw /app/outputs /tmp/matplotlib \
    && chown -R 10001:10001 /app/data /app/outputs /tmp/matplotlib

USER 10001:10001

ENTRYPOINT ["python", "-m", "pln_fx"]
CMD ["--help"]
