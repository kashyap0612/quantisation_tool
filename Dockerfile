FROM python:3.11-slim

# Install system dependencies (some vision/ML libraries require these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# (Hugging Face Spaces run as a non-root user)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install them
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . .

# Expose the port Hugging Face Spaces expects (7860)
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
