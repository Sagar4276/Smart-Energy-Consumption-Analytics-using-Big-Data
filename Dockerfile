# Base image: Java + Python for Spark and Streamlit
FROM openjdk:17-slim

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV JAVA_HOME=/usr/local/openjdk-17
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure directories exist before copying
RUN mkdir -p /app/scripts /app/data /app/model /app/visualization

# Copy only if the directories exist (Docker-native way)
COPY scripts/ /app/scripts/
COPY visualization/ /app/visualization/
# For optional folders — use a .dockerignore trick
# Create empty placeholders if missing
RUN mkdir -p /app/data /app/model

# Expose Streamlit port
EXPOSE 8501

# Default command (idle container)
CMD ["tail", "-f", "/dev/null"]
