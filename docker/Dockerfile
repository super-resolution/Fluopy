# Use an official image for a Python runtime that is based on debian
ARG PYTHON_VERSION=3.14
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-trixie-slim

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

LABEL maintainer="FluopyDevelopers" \
      project="fluopy" \
      python_version=${PYTHON_VERSION}

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot && \
    useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /fluopy

# Volume for data
RUN mkdir /shared && chown nonroot:nonroot /fluopy /shared
VOLUME /shared

# Install locked dependencies before copying the source for better layer caching
COPY --chown=nonroot:nonroot pyproject.toml uv.lock ./

# Use the non-root user for the environment and application
USER nonroot

RUN uv sync --locked --no-default-groups --group test --no-install-project

# Copy and install the project without changing the lockfile
COPY --chown=nonroot:nonroot . .
RUN uv sync --locked --no-default-groups --group test

# Place executables in the environment at the front of the path
ENV PATH="/fluopy/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run a command when the container launches
CMD ["pytest", "-s"]
