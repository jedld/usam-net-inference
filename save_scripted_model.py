#!/usr/bin/env python3

import torch
import os
from model import SAStereoCNN2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model = SAStereoCNN2(device)
    
    # Load the checkpoint
    checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return 1
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move model and all its submodules to device
    model = model.to(device)
    for module in model.modules():
        module.to(device)
    
    # Set to eval mode
    model.eval()
    
    # Create a dummy input for tracing
    print("Creating dummy input for tracing...")
    dummy_left = torch.randn(1, 3, 400, 879, device=device)
    dummy_right = torch.randn(1, 3, 400, 879, device=device)
    dummy_input = torch.cat((dummy_left, dummy_right), dim=1)
    
    # Script the model
    print("Tracing the model...")
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, dummy_input)
    
    # Save the scripted model
    output_path = 'model_scripted.pt'
    print(f"Saving scripted model to {output_path}")
    scripted_model.save(output_path)
    
    print("Done!")
    return 0

if __name__ == "__main__":
    exit(main()) 