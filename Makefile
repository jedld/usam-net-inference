.PHONY: all clean build docker run_docker benchmark

# Default target
all: build

# Build the CUDA extension
build:
	python setup.py install

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -f stereo_cnn_cuda*.so

# Build Docker image
docker:
	docker build -t stereo-cnn-cuda -f Dockerfile.cuda .

# Run Docker container
run_docker:
	docker run --gpus all -v $(PWD):/data stereo-cnn-cuda cuda /data/$(LEFT) /data/$(RIGHT) --output /data/$(OUTPUT) --benchmark

# Run benchmark
benchmark:
	python benchmark.py $(LEFT) $(RIGHT) --iterations $(or $(ITERATIONS),10) --warmup $(or $(WARMUP),2) --models $(or $(MODELS),"baseline stereoRT cuda")

# Help
help:
	@echo "Available targets:"
	@echo "  all         : Build the CUDA extension (default)"
	@echo "  build       : Build the CUDA extension"
	@echo "  clean       : Remove build artifacts"
	@echo "  docker      : Build the Docker image"
	@echo "  run_docker  : Run the Docker container"
	@echo "  benchmark   : Run benchmark"
	@echo ""
	@echo "Variables:"
	@echo "  LEFT        : Path to left image (required for run_docker and benchmark)"
	@echo "  RIGHT       : Path to right image (required for run_docker and benchmark)"
	@echo "  OUTPUT      : Output path for disparity map (default: output.png)"
	@echo "  ITERATIONS  : Number of benchmark iterations (default: 10)"
	@echo "  WARMUP      : Number of warmup iterations (default: 2)"
	@echo "  MODELS      : Models to benchmark (default: baseline stereoRT cuda)"
	@echo ""
	@echo "Example:"
	@echo "  make benchmark LEFT=sample/left.png RIGHT=sample/right.png ITERATIONS=20 MODELS=\"baseline cuda\"" 