# syntax=docker/dockerfile:1

FROM python:3.14-slim

# Install uv (fast Python package manager) by copying its binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first so Docker can cache this layer
# and skip reinstalling deps when only app code changes
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as pinned in uv.lock, no dev dependencies
RUN uv sync --frozen --no-dev

# Now copy the actual application code
COPY app.py config.py ingest.py ./

# Streamlit config for running inside a container
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
