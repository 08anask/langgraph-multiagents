# Dockerfile

# Use the official Python 3.10 image as the base
# We specify 3.10 to match your request for "python 10" (assuming 3.10)
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install 'uv' - a fast Python package installer and resolver
# We use a multi-stage build or directly install it if it's a small binary
# For simplicity, we'll install it directly here.
# You might want to get the latest version from uv's GitHub releases page
# or use pip to install it if available in the Python package index.
# As of my last update, uv is typically installed via pip or standalone binary.
# Let's install it via pip for consistency with Python environment.
RUN pip install uv

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies using uv
# We add --system to install directly into the container's Python environment
# as a virtual environment is not strictly necessary inside a Docker image.
RUN uv pip install --system -r requirements.txt

# Copy the FastAPI application code into the container
COPY app.py .

# Expose port 8000, which is the default port Uvicorn runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# The --host 0.0.0.0 makes the server accessible from outside the container
# The --port 8000 specifies the port
# The --reload flag is useful for development but should be removed for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
