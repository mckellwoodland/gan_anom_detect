# Base image
FROM python:3

# Set up workspace
WORKDIR /workspace
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
