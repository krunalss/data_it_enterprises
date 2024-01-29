# Use the same base image
FROM python:3.8-slim-buster

# Install system dependencies required for Poetry
RUN apt update -y && apt install -y curl awscli

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy only necessary files for installing dependencies
# This helps with Docker caching layers
WORKDIR /app
COPY pyproject.toml poetry.lock /app/

# Disable creation of virtual environments by poetry
# and install dependencies based on the lock file
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application code
COPY . /app

# Set the command to run your application
CMD ["python3", "app.py"]
