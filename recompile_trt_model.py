#!/usr/bin/env python3
import torch
import torch_tensorrt
from model import SAStereoCNN2
import os

def main():
    print("Recompiling TensorRT model with current environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"TensorRT version: {torch_tensorrt.__version__}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available. TensorRT requires CUDA.")
        return

    # Path to checkpoint
    checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Load model
    print("Loading model...")
    model = SAStereoCNN2(device)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare dummy input (same dimensions as in export_to_trt.py)
    print("Creating dummy input for model tracing...")
    # Create half precision input tensors to match the expected input type
    dummy_left = torch.randn(1, 3, 400, 879, device=device, dtype=torch.float32)
    dummy_right = torch.randn(1, 3, 400, 879, device=device, dtype=torch.float32)
    dummy_input = torch.cat((dummy_left, dummy_right), dim=1)
    
    # First trace the model with float32 inputs
    print("Tracing model with float32 inputs...")
    scripted_model = torch.jit.trace(model, dummy_input)

    # Save the scripted model
    scripted_model.save("stereo_cnn_new.ts")
    print("Saved scripted model to stereo_cnn_new.ts")

    # Compile with TensorRT for half precision
    print("Compiling with TensorRT for half precision...")
    try:
        trt_model = torch_tensorrt.compile(
            scripted_model,
            ir="torchscript",
            inputs=[torch_tensorrt.Input(
                (1, 6, 400, 879), 
                dtype=torch.half,  # Specify half precision here
                format=torch.contiguous_format
            )],
            enabled_precisions={torch.half},  # Enable only half precision
            require_full_compilation=True,
            truncate_long_and_double=True
        )

        # Save optimized TRT model
        trt_model.save("model_trt_32_new.ts")
        print("Successfully compiled and saved TensorRT model to model_trt_32_new.ts")
    except Exception as e:
        print(f"Error compiling TensorRT model: {e}")
        return

if __name__ == "__main__":
    main() 