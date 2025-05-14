#!/usr/bin/env python3

import argparse
import torch
import cv2
import numpy as np
from model import SAStereoCNN2
from model_trt import StereoRT
import time
import psutil
import os
import gc

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_video_stream(model_type, video_device, output_path=None, benchmark=False):
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

    model_load_time = time.time() - model_load_start
    model_memory = get_memory_usage() - initial_memory
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_device)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video device {video_device}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Video stream: {width}x{height} @ {fps}fps")
    
    frame_count = 0
    total_inference_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Split frame into left and right images
            # Assuming the frame is side-by-side stereo
            mid = width // 2
            left_img = frame[:, :mid]
            right_img = frame[:, mid:]
            
            # Generate disparity map
            inference_start = time.time()
            if model_type == 'baseline':
                with torch.no_grad():
                    disparity, _ = model.inference(left_img, right_img)
            elif model_type == 'stereoRT':
                with torch.no_grad():
                    disparity = model.inference(left_img, right_img)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Convert disparity to numpy and normalize for visualization
            disparity_np = disparity.squeeze().cpu().numpy()
            if disparity_np.max() > disparity_np.min():
                disparity_np = (disparity_np - disparity_np.min()) / (disparity_np.max() - disparity_np.min())
            
            # Convert to 8-bit color image
            disparity_color = (disparity_np * 255).astype(np.uint8)
            disparity_color = cv2.applyColorMap(disparity_color, cv2.COLORMAP_MAGMA)
            
            # Display results
            cv2.imshow('Left Image', left_img)
            cv2.imshow('Right Image', right_img)
            cv2.imshow('Disparity Map', disparity_color)
            
            # Save frame if output path is provided
            if writer:
                writer.write(disparity_color)
            
            frame_count += 1
            
            # Print benchmark info every 30 frames
            if benchmark and frame_count % 30 == 0:
                avg_inference_time = total_inference_time / frame_count
                print(f"\nBenchmark Results (last 30 frames):")
                print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms")
                print(f"FPS: {1/avg_inference_time:.2f}")
                if torch.cuda.is_available():
                    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
                    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        if benchmark:
            print("\nFinal Benchmark Results:")
            print(f"Total Frames Processed: {frame_count}")
            print(f"Average Inference Time: {(total_inference_time/frame_count)*1000:.2f} ms")
            print(f"Average FPS: {frame_count/total_inference_time:.2f}")
            print(f"Model Memory Usage: {model_memory:.2f} MB")
            print(f"Peak Memory Usage: {get_memory_usage() - initial_memory:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Process stereo video stream through stereo model')
    parser.add_argument('model_type', help='Model type: stereoRt or baseline')
    parser.add_argument('--device', '-d', default='/dev/video0', help='Video device path (default: /dev/video0)')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Show benchmarking information')
    
    args = parser.parse_args()
    
    try:
        process_video_stream(args.model_type, args.device, args.output, args.benchmark)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 