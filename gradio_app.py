import gradio as gr
import torch
import cv2
import numpy as np
from model import SAStereoCNN2
import os
from transforms import test_transform_fn
from PIL import Image

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAStereoCNN2(device)
model.to(device)
CHECKPOINT_PATH = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
if os.path.exists(CHECKPOINT_PATH):
    print("loading existing checkpoint ...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint)
model.eval()

def process_stereo_images(left_img, right_img):
    try:
        # Convert PIL images to numpy arrays
        left_img = np.array(left_img)
        right_img = np.array(right_img)
        
        # Convert to RGB if grayscale
        if len(left_img.shape) == 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB)
        
        # Transform images
        left_tensor = test_transform_fn(left_img).unsqueeze(0).to(device)
        right_tensor = test_transform_fn(right_img).unsqueeze(0).to(device)
        
        # Generate disparity map
        with torch.no_grad():
            disparity = model(left_tensor, right_tensor)
            disparity = disparity.squeeze().cpu().numpy()
        
        # Normalize disparity map for visualization
        disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
        disparity = (disparity * 255).astype(np.uint8)
        
        # Convert to PIL Image for Gradio
        disparity_img = Image.fromarray(disparity)
        
        return disparity_img
    
    except Exception as e:
        raise gr.Error(f"Error processing images: {str(e)}")

# Create Gradio interface
iface = gr.Interface(
    fn=process_stereo_images,
    inputs=[
        gr.Image(label="Left Image", type="pil"),
        gr.Image(label="Right Image", type="pil")
    ],
    outputs=gr.Image(label="Disparity Map"),
    title="Stereo Vision Disparity Map Generator",
    description="Upload left and right stereo images to generate a disparity map.",
    examples=[
        ["examples/left1.jpg", "examples/right1.jpg"],
        ["examples/left2.jpg", "examples/right2.jpg"]
    ] if os.path.exists("examples") else None
)

if __name__ == "__main__":
    iface.launch() 