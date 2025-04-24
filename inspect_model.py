#!/usr/bin/env python3
import torch
import argparse
from model import SAStereoCNN2
from fixed_model import FixedSAStereoCNN2
import numpy as np

def inspect_checkpoint(checkpoint_path):
    """Inspect the model architecture from a checkpoint file"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print parameter shapes
    print("\nParameter shapes in checkpoint:")
    print("=" * 80)
    print(f"{'Parameter':<50} {'Shape':<30}")
    print("-" * 80)
    
    for name, param in checkpoint.items():
        print(f"{name:<50} {str(param.shape):<30}")
    
    print("\nTotal parameters:", sum(p.numel() for p in checkpoint.values()))
    
    return checkpoint

def compare_models(checkpoint_path):
    """Compare the original and fixed model architectures"""
    print(f"Comparing original and fixed models using checkpoint from {checkpoint_path}")
    
    # Create both models
    device = torch.device('cpu')
    original_model = SAStereoCNN2(device)
    fixed_model = FixedSAStereoCNN2(device)
    
    # Load checkpoint into original model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    original_model.load_state_dict(checkpoint)
    
    # Print original model's parameter shapes
    print("\nOriginal model parameter shapes:")
    print("=" * 80)
    print(f"{'Parameter':<50} {'Shape':<30}")
    print("-" * 80)
    
    for name, param in original_model.named_parameters():
        print(f"{name:<50} {str(param.shape):<30}")
    
    # Print fixed model's parameter shapes
    print("\nFixed model parameter shapes:")
    print("=" * 80)
    print(f"{'Parameter':<50} {'Shape':<30}")
    print("-" * 80)
    
    for name, param in fixed_model.named_parameters():
        print(f"{name:<50} {str(param.shape):<30}")
    
    # Find differences
    print("\nParameter shape differences:")
    print("=" * 80)
    print(f"{'Parameter':<50} {'Original Shape':<30} {'Fixed Shape':<30}")
    print("-" * 80)
    
    orig_params = {n: p for n, p in original_model.named_parameters()}
    fixed_params = {n: p for n, p in fixed_model.named_parameters()}
    
    for name in set(orig_params.keys()) & set(fixed_params.keys()):
        if orig_params[name].shape != fixed_params[name].shape:
            print(f"{name:<50} {str(orig_params[name].shape):<30} {str(fixed_params[name].shape):<30}")
    
    # Check if shapes match after loading the checkpoint
    try:
        missing, unexpected = fixed_model.load_state_dict(checkpoint, strict=False)
        print("\nAfter loading checkpoint into fixed model:")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        else:
            print("No missing or unexpected keys - models are compatible!")
    except Exception as e:
        print(f"Error loading checkpoint into fixed model: {e}")

def test_inference(checkpoint_path, input_shape=None):
    """Test model inference with random input"""
    device = torch.device('cpu')
    
    # Create and load the original model
    print("Testing original model...")
    model = SAStereoCNN2(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Create and load the fixed model
    print("Testing fixed model...")
    fixed_model = FixedSAStereoCNN2(device)
    try:
        fixed_model.load_from_original_checkpoint(checkpoint_path)
        fixed_model.eval()
    except Exception as e:
        print(f"Error loading fixed model: {e}")
        return
    
    # Default input shape if not provided
    if input_shape is None:
        height, width = 384, 512
    else:
        height, width = input_shape
    
    print(f"\nTesting inference with input shape: (1, 6, {height}, {width})")
    
    # Generate random input
    x = torch.randn(1, 6, height, width)
    
    # Test original model
    try:
        with torch.no_grad():
            out_orig = model(x)
        print(f"Original model output shape: {out_orig.shape}")
    except Exception as e:
        print(f"Error with original model inference: {e}")
    
    # Test fixed model
    try:
        with torch.no_grad():
            out_fixed = fixed_model(x)
        print(f"Fixed model output shape: {out_fixed.shape}")
    except Exception as e:
        print(f"Error with fixed model inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect model architecture and shapes")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint file")
    parser.add_argument("--mode", choices=["inspect", "compare", "test"], default="compare", 
                        help="Operation mode: inspect, compare, or test")
    
    args = parser.parse_args()
    
    if args.mode == "inspect":
        inspect_checkpoint(args.checkpoint)
    elif args.mode == "compare":
        compare_models(args.checkpoint)
    elif args.mode == "test":
        test_inference(args.checkpoint) 