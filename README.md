# Stereo Depth Estimation with Self-Attention

This project implements a stereo depth estimation system using a deep learning approach with self-attention mechanisms. The system processes stereo image pairs to generate depth maps, providing 3D understanding of scenes from 2D images.

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
- ResNet18 backbone
- Self-attention mechanisms for enhanced feature extraction
- Segment Anything Model (SAM) integration
- Custom stereo matching network

## License

[Add your license information here]

## Acknowledgments

- Facebook Research for the Segment Anything Model
- PyTorch team for the deep learning framework
- Gradio team for the web interface framework
