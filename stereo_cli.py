#!/usr/bin/env python3

import argparse
import torch
import cv2
import numpy as np
from model import SAStereoCNN2
from transforms import test_transform_fn
import matplotlib.pyplot as plt
import os
import time
import psutil
import gc
try:
    from model_trt import StereoRT
    import torch_tensorrt
    print("TensorRT model available")
except ImportError:
    print("TensorRT model not available")
    StereoRT = None


def print_cuda_properties():
    """Print important CUDA properties if CUDA is available"""
    if torch.cuda.is_available():
        print("\nCUDA Properties:")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
        
        # Get detailed device properties
        props = torch.cuda.get_device_properties(0)
        print("\nDetailed Device Properties:")
        print(f"Total Memory: {props.total_memory / 1024**2:.2f} MB")
        print(f"Multi-Processor Count: {props.multi_processor_count}")
        print(f"Max Threads Per Block: {props.max_threads_per_block}")
        print(f"Max Threads Per Multi-Processor: {props.max_threads_per_multi_processor}")
        print(f"Max Block Dimensions: {props.max_block_dim_x} x {props.max_block_dim_y} x {props.max_block_dim_z}")
        print(f"Max Grid Dimensions: {props.max_grid_dim_x} x {props.max_grid_dim_y} x {props.max_grid_dim_z}")
        print(f"Warp Size: {props.warp_size}")
        print(f"Clock Rate: {props.clock_rate / 1000:.2f} GHz")
        print(f"Memory Clock Rate: {props.memory_clock_rate / 1000:.2f} GHz")
        print(f"Memory Bus Width: {props.memory_bus_width} bits")
        print(f"L2 Cache Size: {props.l2_cache_size / 1024:.2f} KB")
        print(f"Compute Mode: {props.compute_mode}")
        print(f"Is Integrated: {props.is_integrated}")
        print(f"Is Multi GPU Board: {props.is_multi_gpu_board}")
        
        print("\nCurrent Memory Status:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        
        # Get current device properties
        print("\nCurrent Device Properties:")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Device Capability: {torch.cuda.get_device_capability()}")
        print(f"Device Properties: {torch.cuda.get_device_properties()}")
    else:
        print("\nCUDA is not available. Running on CPU.")

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_stereo_pair(model_type, left_img_path, right_img_path, output_path='output.png', benchmark=False):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory usage before model loading
    initial_memory = get_memory_usage()
    
    # Model loading time
    model_load_start = time.time()
    if model_type == 'baseline':
        model = SAStereoCNN2(device)
        model.to(device)
        
        # Load checkpoint
        checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
        if os.path.exists(checkpoint_path):
            print("Loading model checkpoint...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
        model.eval()
    elif model_type == 'stereoRT':
        model = StereoRT('model_trt_32.ts')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_load_time = time.time() - model_load_start
    model_memory = get_memory_usage() - initial_memory
    
    # Read and preprocess images
    preprocess_start = time.time()
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        raise ValueError("Failed to load one or both input images")

    preprocess_time = time.time() - preprocess_start
    
    # Generate disparity map
    inference_start = time.time()
    disparity = None
    try:
        if model_type == 'baseline':
            with torch.no_grad():
                disparity, _ = model.inference(left_img, right_img)
        elif model_type == 'stereoRT':
            with torch.no_grad():
                disparity = model.inference(left_img, right_img)
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {str(e)}")
    
    if disparity is None:
        raise RuntimeError("Model inference failed to produce disparity map")
        
    inference_time = time.time() - inference_start
    
    # Post-processing time
    postprocess_start = time.time()
    # Convert disparity to numpy and normalize for visualization
    disparity_np = disparity.squeeze().cpu().numpy()
    if disparity_np.max() > disparity_np.min():
        disparity_np = (disparity_np - disparity_np.min()) / (disparity_np.max() - disparity_np.min())
    
    # Create colormap visualization
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity_np, cmap='magma')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    postprocess_time = time.time() - postprocess_start
    
    # Final memory usage
    final_memory = get_memory_usage()
    peak_memory = final_memory - initial_memory
    
    print(f"Disparity map saved to {output_path}")
    
    if benchmark:
        print("\nBenchmark Results:")
        print(f"Model Loading Time: {model_load_time:.2f} seconds")
        print(f"Preprocessing Time: {preprocess_time:.2f} seconds")
        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"Post-processing Time: {postprocess_time:.2f} seconds")
        print(f"Total Processing Time: {(model_load_time + preprocess_time + inference_time + postprocess_time):.2f} seconds")
        print(f"Model Memory Usage: {model_memory:.2f} MB")
        print(f"Peak Memory Usage: {peak_memory:.2f} MB")
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    
    return disparity_np

def main():
    parser = argparse.ArgumentParser(description='Generate disparity map from stereo image pair')
    parser.add_argument('model_type', help='Model type: stereoRt or baseline')
    parser.add_argument('left_img', help='Path to the left image')
    parser.add_argument('right_img', help='Path to the right image')
    parser.add_argument('--output', '-o', default='output.png', help='Output path for the disparity map (default: output.png)')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Show benchmarking information')
    
    args = parser.parse_args()
    
    # Print CUDA properties at the start
    print_cuda_properties()
    
    try:
        process_stereo_pair(args.model_type, args.left_img, args.right_img, args.output, args.benchmark)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 