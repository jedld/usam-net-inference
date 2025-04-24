# CUDA Implementation of SAStereoCNN2

This is a CUDA C++ implementation of the SAStereoCNN2 model for faster stereo disparity map generation. The implementation uses CUDA kernels for all the major operations in the model, which should provide significant speedups compared to the PyTorch implementation.

## Requirements

- CUDA-enabled GPU
- CUDA Toolkit 11.x
- PyTorch 1.7.0 or later
- OpenCV
- Matplotlib
- Python 3.6 or later

## Installation

### Option 1: Building from Source

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a TorchScript model (required for the CUDA implementation):
   ```
   python save_scripted_model.py --output model_scripted.pt
   ```

3. Install the CUDA extension:
   ```
   python setup.py install
   ```

4. Test the installation:
   ```
   python stereo_cli.py cuda <left-image> <right-image> --output disparity.png --benchmark
   ```

### Option 2: Using Docker

1. Build the Docker image:
   ```
   docker build -t stereo-cnn-cuda -f Dockerfile.cuda .
   ```

2. Run the container:
   ```
   docker run --gpus all -v $(pwd):/data stereo-cnn-cuda cuda /data/<left-image> /data/<right-image> --output /data/disparity.png --benchmark
   ```

## TorchScript Model Conversion

The CUDA implementation requires a TorchScript model to work efficiently. You can convert your PyTorch model checkpoint to TorchScript format using one of the provided scripts:

### Using save_scripted_model.py (Recommended)

This script creates a TorchScript model using `torch.jit.script`:

```bash
python save_scripted_model.py --output model_scripted.pt
```

### Using convert_to_torchscript.py

Alternatively, you can use this script to convert an existing checkpoint:

```bash
python convert_to_torchscript.py --input stereo_cnn_stereo_cnn_sa_baseline.checkpoint --output model_torchscript.pt
```

The CUDA wrapper will automatically look for a TorchScript model named `model_scripted.pt` if it exists, or you can specify a TorchScript model path directly.

## Usage

### Command-Line Interface

The `stereo_cli.py` script has been updated to include the CUDA implementation as an option:

```
python stereo_cli.py cuda <left-image> <right-image> --output disparity.png --benchmark
```

Options:
- `cuda`: Use the CUDA implementation
- `<left-image>`: Path to the left stereo image
- `<right-image>`: Path to the right stereo image
- `--output`: Output path for the disparity map (default: output.png)
- `--benchmark`: Show benchmarking information

### Python API

You can also use the CUDA implementation directly in your Python code:

```python
from stereo_cnn_cuda_wrapper import StereoCNNCuda
import cv2

# Initialize the model (this will automatically use the TorchScript model if available)
model = StereoCNNCuda(checkpoint_path='stereo_cnn_stereo_cnn_sa_baseline.checkpoint')

# Or initialize with a specific TorchScript model
# model = StereoCNNCuda(checkpoint_path='model_scripted.pt')

# Load input images
left_img = cv2.imread('left.png')
right_img = cv2.imread('right.png')

# Run inference
disparity, _ = model.inference(left_img, right_img)

# Convert to numpy for visualization
import torch
import matplotlib.pyplot as plt
import numpy as np

disp_np = disparity.squeeze().cpu().numpy()
if disp_np.max() > disp_np.min():
    disp_np = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min())

plt.figure(figsize=(10, 5))
plt.imshow(disp_np, cmap='magma')
plt.axis('off')
plt.savefig('disparity.png', bbox_inches='tight', pad_inches=0)
```

## Benchmarking

You can benchmark the CUDA implementation against the baseline PyTorch implementation:

```
# Benchmark the CUDA implementation
python stereo_cli.py cuda <left-image> <right-image> --benchmark

# Benchmark the baseline PyTorch implementation
python stereo_cli.py baseline <left-image> <right-image> --benchmark

# Benchmark the TensorRT implementation
python stereo_cli.py stereoRT <left-image> <right-image> --benchmark
```

## Implementation Details

The CUDA implementation consists of the following components:

1. **stereo_cnn_cuda.h/cpp**: C++ wrapper for the model that interfaces with PyTorch
2. **cuda_kernels.cu**: CUDA kernels for all the model operations
3. **stereo_cnn_cuda_wrapper.py**: Python wrapper for the C++/CUDA implementation
4. **setup.py**: Build script for the CUDA extension
5. **save_scripted_model.py**: Script to create a TorchScript model
6. **convert_to_torchscript.py**: Script to convert a checkpoint to TorchScript

The implementation can work in two modes:
- **TorchScript Mode**: Directly uses the TorchScript model for inference (recommended)
- **Manual Implementation**: Uses custom CUDA kernels for all operations

The implementation uses custom CUDA kernels for:
- 2D convolution
- 2D transposed convolution
- BatchNorm2d
- LeakyReLU
- Self-attention
- Sigmoid

## Performance

The CUDA implementation should provide significant speedups compared to the PyTorch implementation, especially for the self-attention mechanism which is computationally intensive. The implementation is optimized for inference and uses fast math CUDA optimizations.

## Troubleshooting

- **Model Loading Errors**: If you encounter errors loading the model, ensure you've created a TorchScript version using either `save_scripted_model.py` or `convert_to_torchscript.py`.
- **TorchScript Compatibility**: The CUDA implementation requires a TorchScript model. Regular PyTorch checkpoints need to be converted first.
- **CUDA Errors**: Ensure your CUDA toolkit is compatible with your PyTorch version.

## Limitations

- Only supports inference (not training)
- Requires a TorchScript model converted from the PyTorch checkpoint
- Limited error handling in the CUDA kernels
- Self-attention implementation is simplified compared to the original 