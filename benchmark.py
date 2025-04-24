#!/usr/bin/env python3

import argparse
import torch
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from model import SAStereoCNN2
from model_trt import StereoRT

# Try to import CUDA implementation
try:
    from stereo_cnn_cuda_wrapper import StereoCNNCuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA extension not available, skipping CUDA benchmarks.")

def load_model(model_type):
    """Load the specified model type"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
    
    if model_type == 'baseline':
        model = SAStereoCNN2(device)
        model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model.eval()
    elif model_type == 'stereoRT':
        model = StereoRT('model_trt_32.ts')
    elif model_type == 'cuda' and CUDA_AVAILABLE:
        model = StereoCNNCuda(checkpoint_path=checkpoint_path, device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def run_benchmark(model_type, left_img_path, right_img_path, iterations=10, warmup=2):
    """Run benchmark for the specified model type"""
    try:
        # Load the model
        model = load_model(model_type)
        
        # Load the images
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None or right_img is None:
            raise ValueError("Failed to load input images")
        
        # Warmup
        print(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            if model_type in ['baseline', 'cuda']:
                disparity, _ = model.inference(left_img, right_img)
            else:
                disparity = model.inference(left_img, right_img)
        
        # Synchronize CUDA operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Run benchmark
        print(f"Running {iterations} benchmark iterations...")
        times = []
        start_time = time.time()
        
        for i in range(iterations):
            iter_start = time.time()
            
            if model_type in ['baseline', 'cuda']:
                disparity, _ = model.inference(left_img, right_img)
            else:
                disparity = model.inference(left_img, right_img)
            
            # Synchronize CUDA operations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            iter_time = time.time() - iter_start
            times.append(iter_time)
            print(f"Iteration {i+1}/{iterations}: {iter_time*1000:.2f} ms")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = np.std(times)
        
        print(f"\nBenchmark Results for {model_type}:")
        print(f"Average Time: {avg_time*1000:.2f} ms")
        print(f"Min Time: {min_time*1000:.2f} ms")
        print(f"Max Time: {max_time*1000:.2f} ms")
        print(f"Standard Deviation: {std_dev*1000:.2f} ms")
        print(f"Total Time for {iterations} iterations: {total_time:.2f} seconds")
        print(f"Throughput: {iterations/total_time:.2f} FPS")
        
        # GPU memory usage
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        
        return {
            'model_type': model_type,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'std_dev': std_dev,
            'total_time': total_time,
            'throughput': iterations/total_time
        }
    
    except Exception as e:
        print(f"Error benchmarking {model_type}: {str(e)}")
        return None

def visualize_results(results):
    """Visualize benchmark results"""
    if not results:
        print("No valid benchmark results to visualize")
        return
    
    # Extract data for plotting
    model_types = [r['model_type'] for r in results if r]
    avg_times = [r['avg_time'] * 1000 for r in results if r]  # Convert to ms
    min_times = [r['min_time'] * 1000 for r in results if r]
    max_times = [r['max_time'] * 1000 for r in results if r]
    throughputs = [r['throughput'] for r in results if r]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot inference times
    bar_width = 0.25
    x = np.arange(len(model_types))
    
    ax1.bar(x - bar_width, avg_times, bar_width, label='Avg Time (ms)')
    ax1.bar(x, min_times, bar_width, label='Min Time (ms)')
    ax1.bar(x + bar_width, max_times, bar_width, label='Max Time (ms)')
    
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Inference Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_types)
    ax1.legend()
    
    # Plot throughput
    ax2.bar(model_types, throughputs)
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Throughput (FPS)')
    ax2.set_title('Throughput Comparison')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()
    
    print(f"Benchmark visualization saved to benchmark_results.png")

def main():
    parser = argparse.ArgumentParser(description='Benchmark stereo model implementations')
    parser.add_argument('left_img', help='Path to the left image')
    parser.add_argument('right_img', help='Path to the right image')
    parser.add_argument('--iterations', '-i', type=int, default=10, help='Number of iterations for benchmark (default: 10)')
    parser.add_argument('--warmup', '-w', type=int, default=2, help='Number of warmup iterations (default: 2)')
    parser.add_argument('--models', '-m', nargs='+', default=['baseline', 'stereoRT', 'cuda'], 
                        help='Models to benchmark (default: all)')
    
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.left_img) or not os.path.exists(args.right_img):
        print(f"Error: Input images not found.")
        return 1
    
    # Run benchmarks
    results = []
    for model_type in args.models:
        if model_type == 'cuda' and not CUDA_AVAILABLE:
            print("Skipping cuda model (not available)")
            continue
            
        print(f"\n{'='*20} Benchmarking {model_type} {'='*20}")
        result = run_benchmark(model_type, args.left_img, args.right_img, args.iterations, args.warmup)
        if result:
            results.append(result)
    
    # Visualize results
    if len(results) > 1:
        visualize_results(results)
    
    return 0

if __name__ == '__main__':
    exit(main()) 