FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy package files and README
COPY pyproject.toml README.md ./

# Install dependencies WITH benchmark extras (matplotlib, pandas)
RUN pip install --no-cache-dir -e ".[benchmark]"

# Copy source code
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/

# Create results directory
RUN mkdir -p benchmarks/results benchmarks/plots

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "benchmarks/run_benchmark.py", "--all"]