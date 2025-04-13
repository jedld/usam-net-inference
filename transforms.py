from torchvision import transforms
from torchvision.transforms import functional as F
import torch
import numpy as np

# IMAGE_SHAPE = (800, 1762)
IMAGE_SHAPE = (400, 879)
IMAGE_SHAPE = (400, 879)

def transform_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize(IMAGE_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.50625424, 0.52283798, 0.41453917], std=[0.21669488, 0.1980729 , 0.18691985])
        ])
def transform_seg_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Resize(IMAGE_SHAPE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

def test_transform_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.50625424, 0.52283798, 0.41453917], std=[0.21669488, 0.1980729 , 0.18691985])
        ])

def test_transform_seg_fn():
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SHAPE, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

def convert_image(image):
    return torch.tensor(image.astype(np.float32) / 256.0).unsqueeze(0)

def transform_disparity_fn():
    return transforms.Compose([
        convert_image,
        transforms.Resize(IMAGE_SHAPE, interpolation=transforms.InterpolationMode.NEAREST)
    ])
