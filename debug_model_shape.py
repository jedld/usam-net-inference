#!/usr/bin/env python3
import torch
import argparse
from model import SAStereoCNN2
import numpy as np

def find_compatible_input_size(checkpoint_path, use_cpu=False):
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
    
    # Try various dimensions until we find one that works
    # The issue is with dimension 3 (width), so we'll try widths that are multiples of 8
    # since the model has 3 downsampling layers (2^3 = 8)
    compatible_sizes = []
    
    # Standard test sizes
    height_options = [480, 384, 320, 240, 224]
    width_options = [640, 576, 512, 448, 416, 384, 352, 320, 256]
    
    print("\nTesting various input sizes for compatibility...")
    
    for height in height_options:
        for width in width_options:
            try:
                print(f"Testing input size: {height}x{width}")
                x = torch.randn(1, 6, height, width, device=device)
                with torch.no_grad():
                    output = model(x)
                print(f"✓ Size {height}x{width} is compatible - Output shape: {output.shape}")
                compatible_sizes.append((height, width))
            except Exception as e:
                print(f"✗ Size {height}x{width} is NOT compatible: {str(e)}")
    
    print("\nCompatible sizes found:")
    for h, w in compatible_sizes:
        print(f"- {h}x{w}")
    
    if compatible_sizes:
        # Choose a compatible size
        chosen_h, chosen_w = compatible_sizes[0]
        print(f"\nProceeding with size {chosen_h}x{chosen_w}")
        
        # Now try to trace the model with the compatible size
        try:
            print("\nTracing model with compatible size...")
            example_input = torch.randn(1, 6, chosen_h, chosen_w, device=device)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save("model_traced_compatible.pt")
            print("Successfully traced and saved model!")
            return True
        except Exception as e:
            print(f"Error tracing model: {e}")
            
            # Try scripting instead
            try:
                print("\nFalling back to scripting...")
                scripted_model = torch.jit.script(model)
                scripted_model.save("model_scripted.pt")
                print("Successfully scripted and saved model!")
                return True
            except Exception as e:
                print(f"Error scripting model: {e}")
                return False
    else:
        print("No compatible sizes found.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug model shape issues and find compatible input sizes")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint file")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU for debugging")
    
    args = parser.parse_args()
    
    success = find_compatible_input_size(args.checkpoint, use_cpu=args.cpu)
    if success:
        print("\nDebugging completed successfully")
    else:
        print("\nDebugging failed to find a solution") 