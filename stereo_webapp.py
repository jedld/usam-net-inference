#!/usr/bin/env python3

import argparse
import torch
import cv2
import numpy as np
from model import SAStereoCNN2
from model_trt import StereoRT
import time
import psutil
import os
import gc
from flask import Flask, Response, render_template
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for video processing
frame_queue = queue.Queue(maxsize=2)
processing_thread = None
stop_event = threading.Event()

def get_memory_usage():
    """Get current memory usage of the process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def process_frames(model, left_cap, right_cap, frame_queue, stop_event):
    """Process frames from both cameras and generate disparity map"""
    frame_count = 0
    total_inference_time = 0
    
    while not stop_event.is_set():
        # Read frames from both cameras
        ret_left, left_frame = left_cap.read()
        ret_right, right_frame = right_cap.read()
        
        if not ret_left or not ret_right:
            logger.error("Failed to read frames from one or both cameras")
            break
        
        # Generate disparity map
        inference_start = time.time()
        with torch.no_grad():
            if isinstance(model, StereoRT):
                disparity = model.inference(left_frame, right_frame)
            else:
                disparity, _ = model.inference(left_frame, right_frame)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # Convert disparity to numpy and normalize for visualization
        disparity_np = disparity.squeeze().cpu().numpy()
        if disparity_np.max() > disparity_np.min():
            disparity_np = (disparity_np - disparity_np.min()) / (disparity_np.max() - disparity_np.min())
        
        # Convert to 8-bit color image
        disparity_color = (disparity_np * 255).astype(np.uint8)
        disparity_color = cv2.applyColorMap(disparity_color, cv2.COLORMAP_MAGMA)
        
        # Create a combined visualization
        h, w = left_frame.shape[:2]
        combined = np.zeros((h, w*3, 3), dtype=np.uint8)
        combined[:, :w] = left_frame
        combined[:, w:w*2] = right_frame
        combined[:, w*2:] = cv2.resize(disparity_color, (w, h))
        
        # Add text labels
        cv2.putText(combined, "Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Right", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Disparity", (w*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (total_inference_time / frame_count)
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Put the frame in the queue
        try:
            frame_queue.put(combined, block=False)
        except queue.Full:
            # If queue is full, remove the oldest frame
            try:
                frame_queue.get_nowait()
                frame_queue.put(combined, block=False)
            except queue.Empty:
                pass

def generate_frames():
    """Generate frames for the video stream"""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            continue

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    parser = argparse.ArgumentParser(description='Stereo video processing web application')
    parser.add_argument('model_type', help='Model type: stereoRt or baseline')
    parser.add_argument('--left-device', default='/dev/video0', help='Left camera device path')
    parser.add_argument('--right-device', default='/dev/video2', help='Right camera device path')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host')
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        if args.model_type == 'baseline':
            model = SAStereoCNN2(device)
            model.to(device)
            
            # Load checkpoint
            checkpoint_path = 'stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
            if os.path.exists(checkpoint_path):
                logger.info("Loading model checkpoint...")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
            
            model.eval()
        elif args.model_type == 'stereoRT':
            model = StereoRT('model_trt_32.ts')
        
        # Initialize video captures
        left_cap = cv2.VideoCapture(args.left_device)
        right_cap = cv2.VideoCapture(args.right_device)
        
        if not left_cap.isOpened() or not right_cap.isOpened():
            raise ValueError("Failed to open one or both cameras")
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=process_frames,
            args=(model, left_cap, right_cap, frame_queue, stop_event)
        )
        processing_thread.start()
        
        # Start Flask app
        logger.info(f"Starting web server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    finally:
        stop_event.set()
        if processing_thread:
            processing_thread.join()
        if left_cap:
            left_cap.release()
        if right_cap:
            right_cap.release()
        cv2.destroyAllWindows()
    
    return 0

if __name__ == '__main__':
    exit(main()) 