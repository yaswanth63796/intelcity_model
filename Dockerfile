# Use Python 3.11 slim image
FROM python:3.11-slim

# Hugging Face Spaces run as a non-root user. 
# We must create a user and give them permissions.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY --chown=user . .

# Expose the port Hugging Face Spaces expects (7860)
EXPOSE 7860

# Run the Flask app with gunicorn on port 7860
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
