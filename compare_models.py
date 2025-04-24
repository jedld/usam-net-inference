#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SAStereoCNN2
from transforms import test_transform_fn
import os
import sys
from stereo_cnn_cuda_wrapper import StereoCNNCuda

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load baseline model
    print("Loading baseline model...")
    baseline_model = SAStereoCNN2(device)
    baseline_model.to(device)
    
    checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        baseline_model.load_state_dict(checkpoint)
    else:
        print(f"Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    
    baseline_model.eval()
    
    # Load CUDA model
    print("Loading CUDA model...")
    cuda_model = StereoCNNCuda(checkpoint_path=checkpoint_path, device=device)
    
    return baseline_model, cuda_model

def compare_outputs(baseline_model, cuda_model, left_img_path, right_img_path):
    # Load images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        print("Failed to load one or both input images")
        sys.exit(1)
    
    # Run baseline model
    print("Running baseline model inference...")
    with torch.no_grad():
        baseline_disparity, _ = baseline_model.inference(left_img, right_img)
    
    # Run CUDA model
    print("Running CUDA model inference...")
    with torch.no_grad():
        cuda_disparity, _ = cuda_model.inference(left_img, right_img)
    
    # Convert to numpy
    baseline_np = baseline_disparity.squeeze().cpu().numpy()
    cuda_np = cuda_disparity.squeeze().cpu().numpy()
    
    # Calculate difference
    diff = np.abs(baseline_np - cuda_np)
    
    # Calculate statistics
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    
    print(f"Mean difference: {mean_diff}")
    print(f"Max difference: {max_diff}")
    print(f"Min difference: {min_diff}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Normalize for visualization
    if baseline_np.max() > baseline_np.min():
        baseline_vis = (baseline_np - baseline_np.min()) / (baseline_np.max() - baseline_np.min())
    else:
        baseline_vis = baseline_np
    
    if cuda_np.max() > cuda_np.min():
        cuda_vis = (cuda_np - cuda_np.min()) / (cuda_np.max() - cuda_np.min())
    else:
        cuda_vis = cuda_np
    
    if diff.max() > diff.min():
        diff_vis = (diff - diff.min()) / (diff.max() - diff.min())
    else:
        diff_vis = diff
    
    # Plot baseline
    plt.subplot(3, 1, 1)
    plt.imshow(baseline_vis, cmap='magma')
    plt.title('Baseline Model Output')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Plot CUDA
    plt.subplot(3, 1, 2)
    plt.imshow(cuda_vis, cmap='magma')
    plt.title('CUDA Model Output')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Plot difference
    plt.subplot(3, 1, 3)
    plt.imshow(diff_vis, cmap='viridis')
    plt.title(f'Absolute Difference (Mean: {mean_diff:.4f}, Max: {max_diff:.4f})')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Saved comparison to model_comparison.png")
    
    # Save individual difference visualization with heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(diff_vis, cmap='hot')
    plt.colorbar(label='Absolute Difference')
    plt.title(f'Difference Map (Mean: {mean_diff:.4f}, Max: {max_diff:.4f})')
    plt.savefig('difference_map.png')
    print("Saved difference map to difference_map.png")
    
    # Optional: Save values arrays for further analysis
    np.save('baseline_disparity.npy', baseline_np)
    np.save('cuda_disparity.npy', cuda_np)
    print("Saved raw disparity arrays to .npy files")
    
    # Additional analysis - histogram of differences
    plt.figure(figsize=(10, 6))
    plt.hist(diff.flatten(), bins=100)
    plt.title('Histogram of Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.savefig('difference_histogram.png')
    print("Saved difference histogram to difference_histogram.png")
    
    # Check for intermediate activations if needed
    # This would require modifying the models to return intermediate outputs
    
    return baseline_np, cuda_np, diff

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline and CUDA model outputs")
    parser.add_argument("--left", default="data/test/left_images/2018-07-11-14-48-52_2018-07-11-14-50-08-769.jpg", 
                        help="Path to left image")
    parser.add_argument("--right", default="data/test/right_images/2018-07-11-14-48-52_2018-07-11-14-50-08-769.jpg", 
                        help="Path to right image")
    
    args = parser.parse_args()
    
    baseline_model, cuda_model = load_models()
    baseline_np, cuda_np, diff = compare_outputs(baseline_model, cuda_model, args.left, args.right)
    
    # Optional: Check for NaN or inf values that might indicate numerical issues
    if np.isnan(cuda_np).any():
        print("Warning: CUDA output contains NaN values!")
    
    if np.isinf(cuda_np).any():
        print("Warning: CUDA output contains infinite values!") 