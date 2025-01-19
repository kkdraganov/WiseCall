# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for the project
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app/

# Install Poetry for dependency management
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry install --no-root

# Expose the port used by FastAPI (default 8000)
EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
