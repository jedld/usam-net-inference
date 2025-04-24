#!/usr/bin/env python3

import torch
import argparse
import os
from model import SAStereoCNN2

def convert_model_to_torchscript(input_checkpoint, output_path):
    """
    Convert a PyTorch model checkpoint to TorchScript format that can be loaded from C++
    
    Args:
        input_checkpoint: Path to the original model checkpoint (.checkpoint or .pth)
        output_path: Path to save the TorchScript model (.pt)
    """
    print(f"Loading model from: {input_checkpoint}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SAStereoCNN2(device)
    model.to(device)
    
    # Load checkpoint
    if os.path.exists(input_checkpoint):
        print("Loading model checkpoint...")
        checkpoint = torch.load(input_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {input_checkpoint}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a dummy input
    dummy_input = torch.randn(1, 6, 384, 1280, device=device)
    
    # Convert model to TorchScript
    print("Converting to TorchScript format...")
    scripted_model = torch.jit.trace(model, dummy_input)
    
    # Save the TorchScript model
    scripted_model.save(output_path)
    print(f"TorchScript model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript format")
    parser.add_argument("--input", required=True, help="Path to the model checkpoint file")
    parser.add_argument("--output", default="model_torchscript.pt", help="Output path for TorchScript model")
    
    args = parser.parse_args()
    
    try:
        convert_model_to_torchscript(args.input, args.output)
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 