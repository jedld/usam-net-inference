# Stereo Depth Estimation with Self-Attention

This project implements a stereo depth estimation system using a deep learning approach with self-attention mechanisms. The system processes stereo image pairs to generate depth maps, providing 3D understanding of scenes from 2D images.

This is a simplified inference only version from:

https://github.com/jedld/usam-net-depth-model

If you wish to train models it is best to look at the above repository instead.

## Features

- Self-attention enhanced stereo depth estimation
- Integration with Segment Anything Model (SAM)
- Real-time depth map generation
- Web-based interface using Gradio
- Support for both CPU and GPU inference

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Gradio
- Segment Anything Model (SAM)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the main SAStereoCNN2 model architecture with self-attention mechanisms
- `gradio_app.py`: Web interface for easy interaction with the model
- `transforms.py`: Image transformation utilities
- `stereo_vision.ipynb`: Jupyter notebook for experimentation and visualization
- `templates/`: Directory containing web interface templates
- `sample/`: Sample stereo image pairs for testing

## Usage

1. Start the web interface:
```bash
python gradio_app.py
```

2. Open your web browser and navigate to the provided local URL
3. Upload a pair of stereo images (left and right views)
4. The system will generate a depth map visualization

## Model Architecture

The model uses a combination of:
- Custom U-Net backbone
- Self-attention mechanisms for enhanced feature extraction
- Segment Anything Model (SAM) integration
- Custom stereo matching network

## Running on the Jetson Orin Nano under Docker 

```
docker build . -t usam-net:latest
```

```
docker run --runtime nvidia -it usam-net 
```

## Training the model from scratch

After downloading the DrivingStereo dataset extract it to the data directory, make sure
that it is of the following structure:

```
── data
│   ├── calib
│   ├── test
│   │   ├── depth_maps
│   │   ├── disparity_maps
│   │   ├── left_images
│   │   ├── left_masks
│   │   ├── right_images
│   │   └── right_masks
│   └── train
│       ├── depth_maps
│       ├── disparity_maps
│       ├── left_images
│       ├── left_masks
│       ├── left_sky_masks
│       ├── right_images
│       └── right_masks
```

## License

MIT License

Copyright (c) 2025 Joseph Emmanuel Dayo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments
- A few sample images from the DrivingStereo dataset https://drivingstereo-dataset.github.io/
- Facebook Research for the Segment Anything Model
- PyTorch team for the deep learning framework
- Gradio team for the web interface framework
