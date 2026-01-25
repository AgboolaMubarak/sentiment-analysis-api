# Slim Python image to keep the container lightweight
FROM python:3.12-slim

# Setting the working directory inside the container
WORKDIR /app

# Install system dependencies  for some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set Environment Variables
# PYTHONPATH ensures 'src' is treated as a package
ENV PYTHONPATH=/app
ENV MODEL_TYPE=advanced

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]