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
from model_trt import StereoRT
import torch_tensorrt

# Add CUDA implementation import
try:
    from stereo_cnn_cuda_wrapper import StereoCNNCuda
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    print("CUDA extension not available. Run 'python setup.py install' to build it.")

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_stereo_pair(model_type, left_img_path, right_img_path, output_path='output.png', benchmark=False, model_path=None, force_checkpoint=False):
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
        # Use specified model path or default
        trt_model_path = model_path if model_path else 'model_trt_32.ts'
        print(f"Loading TensorRT model from: {trt_model_path}")
        model = StereoRT(trt_model_path)
    elif model_type == 'cuda' and CUDA_EXTENSION_AVAILABLE:
        checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        model = StereoCNNCuda(checkpoint_path=checkpoint_path, device=device, force_checkpoint=force_checkpoint)
    else:
        if model_type == 'cuda' and not CUDA_EXTENSION_AVAILABLE:
            raise ImportError("CUDA extension not available. Run 'python setup.py install' to build it.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'baseline', 'stereoRT', or 'cuda'.")

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
    if model_type in ['baseline', 'cuda']:
        with torch.no_grad():
            disparity, _ = model.inference(left_img, right_img)
    elif model_type == 'stereoRT':
        with torch.no_grad():
            disparity = model.inference(left_img, right_img)
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
        print(f"Model Type: {model_type}")
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
    parser.add_argument('model_type', help='Model type: stereoRT, baseline, or cuda')
    parser.add_argument('left_img', help='Path to the left image')
    parser.add_argument('right_img', help='Path to the right image')
    parser.add_argument('--output', '-o', default='output.png', help='Output path for the disparity map (default: output.png)')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Show benchmarking information')
    parser.add_argument('--model-path', type=str, help='Custom path to the TensorRT model file (for stereoRT)')
    parser.add_argument('--force-checkpoint', action='store_true', help='Force CUDA model to use checkpoint file directly')
    
    args = parser.parse_args()
    
    try:
        process_stereo_pair(args.model_type, args.left_img, args.right_img, args.output, args.benchmark, args.model_path, args.force_checkpoint)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 