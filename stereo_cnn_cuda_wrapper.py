import torch
import torch.nn as nn
import stereo_cnn_cuda
from transforms import test_transform_fn
import cv2
import os

class StereoCNNCuda(nn.Module):
    def __init__(self, checkpoint_path=None, device='cuda', force_checkpoint=False):
        super(StereoCNNCuda, self).__init__()
        
        self.device = device
        self.model = stereo_cnn_cuda.StereoModelCUDA()
        
        # Set default TorchScript model path
        torchscript_path = 'model_scripted.pt'
        
        if os.path.exists(torchscript_path):
            print(f"Loading TorchScript model: {torchscript_path}")
            self.model.load_weights(torchscript_path)
        else:
            print(f"TorchScript model not found at {torchscript_path}. Please run 'python save_scripted_model.py' first.")
            # If a checkpoint is provided and force_checkpoint is true, attempt to load it
            if checkpoint_path and force_checkpoint:
                print(f"Attempting to load checkpoint directly: {checkpoint_path}")
                self.model.load_weights(checkpoint_path)
            else:
                raise FileNotFoundError(f"TorchScript model {torchscript_path} not found and force_checkpoint not set.")
    
    def forward(self, x):
        # Ensure input is on the right device
        x = x.to(self.device)
        return self.model.forward(x)
    
    def inference(self, left_img, right_img):
        """
        Process a pair of images to produce a disparity map
        
        Args:
            left_img: Left RGB image (numpy array, HxWx3)
            right_img: Right RGB image (numpy array, HxWx3)
        
        Returns:
            disparity: Disparity map as a tensor
        """
        # Apply image transformations
        transform = test_transform_fn()
        left_tensor = transform(left_img).unsqueeze(0).to(self.device)
        right_tensor = transform(right_img).unsqueeze(0).to(self.device)
        
        # Concatenate images for model input
        input_tensor = torch.cat((left_tensor, right_tensor), dim=1)
        
        # Run inference
        with torch.no_grad():
            disparity = self.forward(input_tensor)
        
        return disparity, None  # Maintain compatibility with original model

# Benchmarking function
def benchmark_cuda_model(model, left_img_path, right_img_path, iterations=10):
    """Benchmark the CUDA model performance"""
    import time
    import numpy as np
    
    # Read test images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        raise ValueError("Failed to load test images")
    
    # Warmup
    model.inference(left_img, right_img)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        disparity, _ = model.inference(left_img, right_img)
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    
    print(f"Average inference time for {iterations} iterations: {avg_time*1000:.2f} ms")
    return avg_time

# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the CUDA Stereo CNN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--left", type=str, required=True, help="Path to left image")
    parser.add_argument("--right", type=str, required=True, help="Path to right image")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output", type=str, default="disparity_cuda.png", help="Output file")
    parser.add_argument("--force-checkpoint", action="store_true", help="Force using the checkpoint file directly")
    
    args = parser.parse_args()
    
    # Initialize model
    model = StereoCNNCuda(checkpoint_path=args.checkpoint, force_checkpoint=args.force_checkpoint)
    
    # Run inference
    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    disparity, _ = model.inference(left_img, right_img)
    
    # Save output
    import matplotlib.pyplot as plt
    disp_np = disparity.squeeze().cpu().numpy()
    
    if disp_np.max() > disp_np.min():
        disp_np = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min())
    
    plt.figure(figsize=(10, 5))
    plt.imshow(disp_np, cmap='magma')
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Saved disparity map to {args.output}")
    
    # Run benchmark if requested
    if args.benchmark:
        avg_time = benchmark_cuda_model(model, args.left, args.right)
        print(f"Benchmark result: {avg_time*1000:.2f} ms per inference") 