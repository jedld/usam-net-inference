#!/usr/bin/env python3
import torch
import argparse
from fixed_model import FixedSAStereoCNN2

def convert_with_fixed_model(checkpoint_path, output_path, use_cpu=False, input_size=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Explicitly set device
    device = torch.device('cpu') if use_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create fixed model
    model = FixedSAStereoCNN2(device)
    
    # Load weights from original checkpoint
    try:
        model.load_from_original_checkpoint(checkpoint_path)
        print(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    # Set model to evaluation mode for inference
    model.eval()
    
    # Default input size that should work with our fixed model
    if input_size is None:
        height, width = 384, 512  # Sizes divisible by 32 to ensure compatibility
    else:
        height, width = input_size
    
    # Enable debug mode for detailed shape information
    model.debug_mode = True
    
    # Test forward pass to verify shapes
    print(f"\nTesting forward pass with input size: {height}x{width}...")
    try:
        x = torch.randn(1, 6, height, width, device=device)
        with torch.no_grad():
            output = model(x)
        print(f"Test forward pass successful - Output shape: {output.shape}")
    except Exception as e:
        print(f"Error in test forward pass: {e}")
        return False
    
    # Disable debug mode for tracing
    model.debug_mode = False
    
    # Try the TorchScript conversion
    try:
        # First try tracing
        print("\nTracing model...")
        example_input = torch.randn(1, 6, height, width, device=device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(output_path)
        print(f"Successfully traced and saved model to {output_path}")
        return True
    except Exception as e:
        print(f"Error tracing model: {e}")
        
        # If tracing fails, try scripting
        try:
            print("\nFalling back to scripting...")
            scripted_model = torch.jit.script(model)
            scripted_output = output_path.replace('.pt', '_scripted.pt')
            scripted_model.save(scripted_output)
            print(f"Successfully scripted and saved model to {scripted_output}")
            return True
        except Exception as e:
            print(f"Error scripting model: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert checkpoint to TorchScript using fixed model")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint file")
    parser.add_argument("--output", "-o", default="fixed_model_torchscript.pt", help="Output path for TorchScript model")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU for conversion")
    parser.add_argument("--height", type=int, help="Input height (default: 384)")
    parser.add_argument("--width", type=int, help="Input width (default: 512)")
    
    args = parser.parse_args()
    
    input_size = None
    if args.height is not None and args.width is not None:
        input_size = (args.height, args.width)
    
    success = convert_with_fixed_model(args.checkpoint, args.output, use_cpu=args.cpu, input_size=input_size)
    if success:
        print("\nConversion completed successfully")
    else:
        print("\nConversion failed") 