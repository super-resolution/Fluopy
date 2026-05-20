# Use an official image for a Python runtime that is based on debian
ARG PYTHON_VERSION=3.14
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-trixie-slim

ARG PYTHON_VERSION

ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV UV_SYSTEM_PYTHON=1
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Omit development dependencies
ENV UV_NO_DEV=1
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

LABEL maintainer="FluopyDevelopers" \
      project="fluopy" \
      python_version=${PYTHON_VERSION}

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot && \
    useradd --system --gid 999 --uid 999 --create-home nonroot

# set time zone to local time
RUN ln -sf /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# install git for setuptools_scm to deal with locan source distribution versioning
# remove package manager cache
RUN apt-get update && \
    apt-get install -yq \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /fluopy

# Copy the current directory contents into the container
COPY . /fluopy

# install dependencies
RUN uv sync --group test --no-install-project

# install project
RUN uv sync --inexact

# Place executables in the environment at the front of the path
ENV PATH="/fluopy/.venv/bin:$PATH"

# Volume for data
VOLUME /shared
RUN mkdir /shared
RUN chown -R nonroot:nonroot /shared

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Use the non-root user to run our application
USER nonroot

# Run a command when the container launches
CMD today=$(date +"%Y-%m-%d") && \
    base="-linux-py"${PYTHON_VERSION}_$today && \
    date > "/shared/test_results$base.txt" && \
    pytest -s >> "/shared/test_results$base.txt"
