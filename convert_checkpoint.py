#!/usr/bin/env python3
import torch
import argparse
from model import SAStereoCNN2

def convert_checkpoint_to_torchscript(checkpoint_path, output_path, use_cpu=False):
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Explicitly set device
    device = torch.device('cpu') if use_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and load weights
    model = SAStereoCNN2(device)
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            # If it's already a state dict
            state_dict = checkpoint
        else:
            # Try to get state dict from checkpoint
            try:
                state_dict = checkpoint.state_dict()
            except:
                print("Warning: Could not extract state_dict from checkpoint, using as-is")
                state_dict = checkpoint
        
        # Load state dict into model
        model.load_state_dict(state_dict)
        
        # Make sure model is on the correct device
        model = model.to(device)
        print(f"Checkpoint loaded successfully and moved to {device}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    # Set model to evaluation mode for inference
    model.eval()
    
    # Create example input for tracing - ensure it's on the same device as the model
    example_input = torch.randn(1, 6, 480, 640, device=device)
    
    # Verify device consistency
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {example_input.device}")
    
    try:
        # Trace the model with example input
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        traced_model.save(output_path)
        print(f"TorchScript model saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error tracing model: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to TorchScript model")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint file")
    parser.add_argument("--output", "-o", default="model_torchscript.pt", help="Output path for TorchScript model")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU for conversion")
    
    args = parser.parse_args()
    
    success = convert_checkpoint_to_torchscript(args.checkpoint, args.output, use_cpu=args.cpu)
    if success:
        print("Conversion successful")
    else:
        print("Conversion failed") 