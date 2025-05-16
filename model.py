import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transforms import transform_disparity_fn, transform_fn, transform_seg_fn, test_transform_fn, test_transform_seg_fn
import cv2

# Try to import TensorRT, but make it optional
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. StereoRT model will not be available.")

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out + x

    def flops(self, x):
        batch_size, channels, height, width = x.size()
        flops = 0
        flops += self.query_conv.weight.numel() * height * width
        flops += self.key_conv.weight.numel() * height * width
        flops += self.value_conv.weight.numel() * height * width
        flops += height * width * (channels // 8) * (height * width)  # For the attention matrix multiplication
        flops += height * width * channels * (height * width)  # For the final matrix multiplication
        return flops

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class SAStereoCNN2(nn.Module):
    def __init__(self, device, load_sam=False):
        super(SAStereoCNN2, self).__init__()
        self.device = device
        
        # Initialize all layers
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256))
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))

        self.self_attention = SelfAttention(1024)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        
        # Move all parameters to device immediately
        self.to(device)
        
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)    
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255
    
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        left_img = transform(left_img).unsqueeze(0).to(self.device)
        right_img = transform(right_img).unsqueeze(0).to(self.device)
        input = torch.cat((left_img, right_img), 1)
        return self.forward(input), None    