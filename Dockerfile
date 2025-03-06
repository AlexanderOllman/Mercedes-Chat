FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary packages first
RUN apt-get update && \
    apt-get install -y curl nano git git-lfs ca-certificates && \
    git lfs install && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install additional dependencies for image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Configure git globally
RUN git config --global --add safe.directory /app

# Create placeholder image directory if it doesn't exist
RUN mkdir -p /app/static/img
RUN touch /app/static/img/placeholder.jpg

# Make port 80 and 8080 available
EXPOSE 80
EXPOSE 8080

# Set the default Git repository URL
ENV GIT_REPO_URL="https://github.com/AlexanderOllman/Mercedes-Chat.git"

# Add to your existing Dockerfile
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0

# Ensure the app has permissions to restart itself
RUN chmod +x /app/app.py

# Run app.py when the container launches
CMD ["python3", "app.py"]
