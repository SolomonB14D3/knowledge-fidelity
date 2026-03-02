FROM python:3.11-slim

WORKDIR /app

# Install system deps for scipy/numpy builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the library
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[align]"

# All 11 CLI tools are now available:
#   rho-eval, rho-audit, rho-surgery, rho-benchmark, rho-bench,
#   rho-compress, rho-interpret, rho-align, rho-steer, rho-hybrid,
#   rho-leaderboard

ENTRYPOINT ["rho-eval"]
CMD ["--help"]
