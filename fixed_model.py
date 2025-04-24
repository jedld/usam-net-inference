import torch
import torch.nn as nn
from model import SelfAttention

class FixedSAStereoCNN2(nn.Module):
    def __init__(self, device):
        super(FixedSAStereoCNN2, self).__init__()
        # Downsampling blocks - using padding='same' to maintain spatial dimensions
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
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )

        self.self_attention = SelfAttention(1024)

        # Upsampling blocks - USING SAME KERNEL SIZES AS ORIGINAL MODEL (3x3)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        
        # Fixed up5 based on original model
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.device = device
        self.to(device)

    def forward(self, x):
        # Store sizes for debugging
        input_size = x.size()
        
        # Downsampling path
        down1 = self.down1(x)
        down1_size = down1.size()
        
        down2 = self.down2(down1)
        down2_size = down2.size()
        
        down3 = self.down3(down2)
        down3_size = down3.size()
        
        down4 = self.down4(down3)
        down4_size = down4.size()
        
        down5 = self.down5(down4)
        down5_size = down5.size()
        
        # Self-attention
        sa = self.self_attention(down5)
        sa_size = sa.size()
        
        # Upsampling path with skip connections
        # For each upsampling step, ensure compatible sizes for addition
        up1 = self.up1(sa)
        # Ensure up1 has same size as down4
        if up1.size(2) != down4.size(2) or up1.size(3) != down4.size(3):
            up1 = nn.functional.interpolate(up1, size=(down4.size(2), down4.size(3)), mode='bilinear', align_corners=False)
        up1 = up1 + down4
        up1_size = up1.size()
        
        up2 = self.up2(up1)
        # Ensure up2 has same size as down3
        if up2.size(2) != down3.size(2) or up2.size(3) != down3.size(3):
            up2 = nn.functional.interpolate(up2, size=(down3.size(2), down3.size(3)), mode='bilinear', align_corners=False)
        up2 = up2 + down3
        up2_size = up2.size()
        
        up3 = self.up3(up2)
        # Ensure up3 has same size as down2
        if up3.size(2) != down2.size(2) or up3.size(3) != down2.size(3):
            up3 = nn.functional.interpolate(up3, size=(down2.size(2), down2.size(3)), mode='bilinear', align_corners=False)
        up3 = up3 + down2
        up3_size = up3.size()
        
        up4 = self.up4(up3)
        # Ensure up4 has same size as down1
        if up4.size(2) != down1.size(2) or up4.size(3) != down1.size(3):
            up4 = nn.functional.interpolate(up4, size=(down1.size(2), down1.size(3)), mode='bilinear', align_corners=False)
        up4 = up4 + down1
        up4_size = up4.size()
        
        up5 = self.up5(up4)
        up5_size = up5.size()
        
        # Final convolution
        out = self.conv(up5)
        out_size = out.size()
        
        # Debugging info - only print when explicitly requested
        if getattr(self, 'debug_mode', False):
            print(f"Input size: {input_size}")
            print(f"Down1 size: {down1_size}")
            print(f"Down2 size: {down2_size}")
            print(f"Down3 size: {down3_size}")
            print(f"Down4 size: {down4_size}")
            print(f"Down5 size: {down5_size}")
            print(f"SA size: {sa_size}")
            print(f"Up1 size: {up1_size}")
            print(f"Up2 size: {up2_size}")
            print(f"Up3 size: {up3_size}")
            print(f"Up4 size: {up4_size}")
            print(f"Up5 size: {up5_size}")
            print(f"Output size: {out_size}")
        
        return out * 255
    
    def load_from_original_checkpoint(self, checkpoint_path):
        """Load weights from original SAStereoCNN2 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # New state dict that will receive the converted weights
        new_state_dict = {}
        
        # Map from original keys to new keys
        for key, value in checkpoint.items():
            # Most layer names are the same, so we can directly copy
            new_state_dict[key] = value
        
        # Load the converted state dict
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"WARNING: Missing keys in checkpoint: {missing_keys}")
        
        if len(unexpected_keys) > 0:
            print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}")
            
        print(f"Loaded weights from {checkpoint_path}")
    
    def inference(self, left_img, right_img):
        from transforms import test_transform_fn
        transform = test_transform_fn()
        left_img = transform(left_img).unsqueeze(0).to(self.device)
        right_img = transform(right_img).unsqueeze(0).to(self.device)
        input = torch.cat((left_img, right_img), 1)
        return self.forward(input), None 