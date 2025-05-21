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
from statistics import mean, stdev
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
        
        print("\nCurrent Memory Status:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        
        # Get current device properties
        print("\nCurrent Device Properties:")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        print(f"Device Capability: {torch.cuda.get_device_capability()}")
    else:
        print("\nCUDA is not available. Running on CPU.")

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(model_type, left_img_path, right_img_path, num_runs=5, cpu_only=False):
    """Run multiple benchmarks and return statistics"""
    timings = {
        'model_load': [],
        'preprocess': [],
        'inference': [],
        'postprocess': [],
        'total': [],
        'model_memory': [],
        'peak_memory': []
    }
    
    # First run to load model and warm up
    print("Warming up...")
    process_stereo_pair(model_type, left_img_path, right_img_path, 'warmup.png', benchmark=False, cpu_only=cpu_only)
    
    print(f"\nRunning {num_runs} benchmark iterations...")
    for i in range(num_runs):
        print(f"\nIteration {i+1}/{num_runs}")
        try:
            # Clear CUDA cache if available and not in CPU-only mode
            if torch.cuda.is_available() and not cpu_only:
                torch.cuda.empty_cache()
            
            # Run benchmark
            start_time = time.time()
            process_stereo_pair(model_type, left_img_path, right_img_path, f'output_{i}.png', benchmark=True, cpu_only=cpu_only)
            total_time = time.time() - start_time
            
            # Parse benchmark output
            with open('benchmark_output.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Model Loading Time:' in line:
                        timings['model_load'].append(float(line.split(':')[1].strip().split()[0]))
                    elif 'Preprocessing Time:' in line:
                        timings['preprocess'].append(float(line.split(':')[1].strip().split()[0]))
                    elif 'Inference Time:' in line:
                        timings['inference'].append(float(line.split(':')[1].strip().split()[0]))
                    elif 'Post-processing Time:' in line:
                        timings['postprocess'].append(float(line.split(':')[1].strip().split()[0]))
                    elif 'Model Memory Usage:' in line:
                        timings['model_memory'].append(float(line.split(':')[1].strip().split()[0]))
                    elif 'Peak Memory Usage:' in line:
                        timings['peak_memory'].append(float(line.split(':')[1].strip().split()[0]))
            
            timings['total'].append(total_time)
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {str(e)}")
            continue
    
    # Calculate statistics
    stats = {}
    for key, values in timings.items():
        if values:  # Only calculate if we have values
            stats[key] = {
                'mean': mean(values),
                'std': stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }
    
    return stats

def print_benchmark_stats(stats):
    """Print benchmark statistics in a formatted table"""
    print("\nBenchmark Statistics:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Mean (s)':<12} {'Std Dev (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
    print("-" * 80)
    
    for key in ['model_load', 'preprocess', 'inference', 'postprocess', 'total']:
        if key in stats:
            print(f"{key:<20} {stats[key]['mean']:<12.3f} {stats[key]['std']:<12.3f} "
                  f"{stats[key]['min']:<12.3f} {stats[key]['max']:<12.3f}")
    
    print("\nMemory Usage Statistics:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Mean (MB)':<12} {'Std Dev (MB)':<12} {'Min (MB)':<12} {'Max (MB)':<12}")
    print("-" * 80)
    
    for key in ['model_memory', 'peak_memory']:
        if key in stats:
            print(f"{key:<20} {stats[key]['mean']:<12.1f} {stats[key]['std']:<12.1f} "
                  f"{stats[key]['min']:<12.1f} {stats[key]['max']:<12.1f}")

def process_stereo_pair(model_type, left_img_path, right_img_path, output_path='output.png', benchmark=False, cpu_only=False, load_checkpoint=True):
    # Initialize model
    if cpu_only:
        device = torch.device('cpu')
        print("Forcing CPU usage as requested")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and model_type != 'base' else 'cpu')
    print(f"Using device: {device}")
    
    # Check if we're running on Jetson Orin
    is_jetson = False
    if device.type == 'cuda':
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major == 8 and device_props.minor == 7:  # Orin Nano
            is_jetson = True
            print("Detected Jetson Orin Nano - applying specific optimizations")
            # Set optimal settings for Jetson
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 as it's not optimal for Orin
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Set lower memory fraction for Jetson's shared memory architecture
            torch.cuda.set_per_process_memory_fraction(0.7)  # More conservative memory usage
            # Enable INT8 optimizations if available
            if hasattr(torch, 'quantization'):
                torch.quantization.observer.default_observer = torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
                )
    
    # Memory usage before model loading
    initial_memory = get_memory_usage()
    
    # Model loading time
    model_load_start = time.time()
    if model_type in ['baseline', 'base']:
        model = SAStereoCNN2(device, load_checkpoint=load_checkpoint)
        model.to(device)
        
        # Enable inference optimizations
        model.eval()
        if device.type == 'cuda':
            if not is_jetson:
                # Original RTX optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.set_per_process_memory_fraction(0.9)
            torch.cuda.empty_cache()
    elif model_type == 'stereoRT':
        if cpu_only:
            raise ValueError("TensorRT model cannot be used in CPU-only mode")
        if StereoRT is None:
            raise ImportError("TensorRT model is not available. Please install required dependencies.")
        model = StereoRT('model_trt_32.ts')
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'base', 'baseline', or 'stereoRT'")

    model_load_time = time.time() - model_load_start
    model_memory = get_memory_usage() - initial_memory
    
    # Read and preprocess images
    preprocess_start = time.time()
    # Use cv2.IMREAD_UNCHANGED for faster loading
    left_img = cv2.imread(left_img_path, cv2.IMREAD_UNCHANGED)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_UNCHANGED)
    
    if left_img is None or right_img is None:
        raise ValueError("Failed to load one or both input images")

    preprocess_time = time.time() - preprocess_start
    
    # Generate disparity map
    inference_start = time.time()
    disparity = None
    try:
        if model_type in ['baseline', 'base']:
            with torch.no_grad():
                if is_jetson:
                    # Use FP16 for Jetson Orin
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        disparity, _ = model.inference(left_img, right_img)
                else:
                    # Original mixed precision for other GPUs
                    with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
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
    
    # Create colormap visualization - use a more efficient approach
    plt.figure(figsize=(10, 5), dpi=100)  # Reduced DPI for faster saving
    plt.imshow(disparity_np, cmap='magma')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Removed optimize parameter
    plt.close()
    postprocess_time = time.time() - postprocess_start
    
    # Final memory usage
    final_memory = get_memory_usage()
    peak_memory = final_memory - initial_memory
    
    print(f"Disparity map saved to {output_path}")
    
    if benchmark:
        # Write benchmark results to a file for parsing
        with open('benchmark_output.txt', 'w') as f:
            f.write(f"Model Loading Time: {model_load_time:.2f} seconds\n")
            f.write(f"Preprocessing Time: {preprocess_time:.2f} seconds\n")
            f.write(f"Inference Time: {inference_time:.2f} seconds\n")
            f.write(f"Post-processing Time: {postprocess_time:.2f} seconds\n")
            f.write(f"Total Processing Time: {(model_load_time + preprocess_time + inference_time + postprocess_time):.2f} seconds\n")
            f.write(f"Model Memory Usage: {model_memory:.2f} MB\n")
            f.write(f"Peak Memory Usage: {peak_memory:.2f} MB\n")
            if torch.cuda.is_available():
                f.write(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB\n")
                f.write(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB\n")
    
    return disparity_np

def main():
    parser = argparse.ArgumentParser(description='Generate disparity map from stereo image pair')
    parser.add_argument('model_type', choices=['base', 'baseline', 'stereoRT'], 
                      help='Model type: base (CPU only), baseline (GPU if available), or stereoRT (TensorRT)')
    parser.add_argument('left_img', help='Path to the left image')
    parser.add_argument('right_img', help='Path to the right image')
    parser.add_argument('--output', '-o', default='output.png', help='Output path for the disparity map (default: output.png)')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Show benchmarking information')
    parser.add_argument('--runs', '-r', type=int, default=5, help='Number of benchmark runs (default: 5)')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU usage even if CUDA is available')
    parser.add_argument('--no-checkpoint', action='store_true', help='Skip loading the model checkpoint')
    
    args = parser.parse_args()
    
    # Print CUDA properties at the start
    print_cuda_properties()
    
    try:
        if args.benchmark:
            stats = run_benchmark(args.model_type, args.left_img, args.right_img, args.runs, args.cpu_only)
            print_benchmark_stats(stats)
        else:
            process_stereo_pair(args.model_type, args.left_img, args.right_img, args.output, args.benchmark, args.cpu_only, not args.no_checkpoint)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 