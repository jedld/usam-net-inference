#!/bin/bash

# Create a directory for profiling results on the host
mkdir -p profiling_results

# Build the Docker image
echo "Building Docker image..."
docker build . -t stereo-cnn-cuda -f Dockerfile.cuda

# Run the container with volume mount for profiling results
echo "Running container with profiling..."
docker run --gpus all \
    -v $(pwd)/profiling_results:/app/profiling_results \
    stereo-cnn-cuda

# Check if profiling was successful
if [ -f "profiling_results/stereo_cnn_profile.nsys-rep" ]; then
    echo "Profiling completed successfully!"
    echo "Results saved to profiling_results/stereo_cnn_profile.nsys-rep"
    
    # Generate summary report
    echo "Generating summary report..."
    nsys stats profiling_results/stereo_cnn_profile.nsys-rep > profiling_results/summary.txt
    
    # Export to CSV
    echo "Exporting to CSV..."
    nsys export --type csv profiling_results/stereo_cnn_profile.nsys-rep
    
    echo "You can view the detailed results using:"
    echo "nsys-ui profiling_results/stereo_cnn_profile.nsys-rep"
else
    echo "Error: Profiling results not found!"
    exit 1
fi 