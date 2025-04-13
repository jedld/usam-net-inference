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

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_stereo_pair(left_img_path, right_img_path, output_path='output.png', benchmark=False):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory usage before model loading
    initial_memory = get_memory_usage()
    
    # Model loading time
    model_load_start = time.time()
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
    
    model_load_time = time.time() - model_load_start
    model_memory = get_memory_usage() - initial_memory
    
    model.eval()
    
    # Read and preprocess images
    preprocess_start = time.time()
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        raise ValueError("Failed to load one or both input images")


    preprocess_time = time.time() - preprocess_start
    
    # Generate disparity map
    inference_start = time.time()
    with torch.no_grad():
        disparity, _ = model.inference(left_img, right_img)
    inference_time = time.time() - inference_start
    
    # Post-processing time
    postprocess_start = time.time()
    # Convert disparity to numpy and normalize for visualization
    disparity_np = disparity.squeeze().cpu().numpy()
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
    parser.add_argument('left_img', help='Path to the left image')
    parser.add_argument('right_img', help='Path to the right image')
    parser.add_argument('--output', '-o', default='output.png', help='Output path for the disparity map (default: output.png)')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Show benchmarking information')
    
    args = parser.parse_args()
    
    try:
        process_stereo_pair(args.left_img, args.right_img, args.output, args.benchmark)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 