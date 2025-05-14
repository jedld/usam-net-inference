FROM dustynv/l4t-pytorch:r36.4.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY stereo_webapp.py .
COPY model.py .
COPY model_trt.py .
COPY stereo_cnn_stereo_cnn_sa_baseline.checkpoint .
COPY model_trt_32.ts .
COPY templates/ templates/

# Create a non-root user
RUN useradd -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python3", "stereo_webapp.py", "stereoRT", "--host", "0.0.0.0", "--port", "5000"]
