#!/usr/bin/env python3

import argparse
import torch
import cv2
from model import SAStereoCNN2
import matplotlib.pyplot as plt
import os

def save_raw_disparity(disparity_np, output_path):
    cv2.imwrite(output_path, disparity_np)

def process_stereo_pair(left_img_path, right_img_path, output_path='output.png', raw=False):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAStereoCNN2(device)
    model.to(device)
    # Load checkpoint
    checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
    if os.path.exists(checkpoint_path):
        print("Loading model checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    model.eval()
    
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    
    with torch.no_grad():
        disparity, _ = model.inference(left_img, right_img)

    
    disparity_np = disparity.squeeze().cpu().numpy()

    if raw:
        save_raw_disparity(disparity_np, output_path)
    else:
    
        plt.figure(figsize=(10, 5))
        plt.imshow(disparity_np, cmap='gray')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"Disparity map saved to {output_path}")
    return disparity_np

def main():
    parser = argparse.ArgumentParser(description='Generate disparity map from stereo image pair')
    parser.add_argument('left_img', help='Path to the left image')
    parser.add_argument('right_img', help='Path to the right image')
    parser.add_argument('--output', '-o', default='output.png', help='Output path for the disparity map (default: output.png)')
    parser.add_argument('--raw', '-r', action='store_true', help='Output raw disparity map (default: False)')
    
    args = parser.parse_args()
    
    try:
        process_stereo_pair(args.left_img, args.right_img, args.output, args.raw)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 