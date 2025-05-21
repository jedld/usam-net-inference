import torch
from transforms import test_transform_fn

class StereoRT:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = test_transform_fn()
        self.stream = torch.cuda.Stream()
        
        # Enable TF32 for better performance on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        # Set optimal memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def _load_model(self, path):
        model = torch.jit.load(path, map_location=self.device)
        model.eval()
        return model

    def _prepare_input(self, left_image, right_image):
        with torch.cuda.stream(self.stream):
            # Use non-blocking transfers and FP16
            with torch.cuda.amp.autocast():
                left_tensor = self.transform(left_image).unsqueeze(0).to(self.device, non_blocking=True)
                right_tensor = self.transform(right_image).unsqueeze(0).to(self.device, non_blocking=True)
                input_tensor = torch.cat((left_tensor, right_tensor), dim=1)

        self.stream.synchronize()
        return input_tensor

    def inference(self, left_image, right_image):
        with torch.no_grad():
            input_tensor = self._prepare_input(left_image, right_image)
            with torch.cuda.stream(self.stream):
                with torch.cuda.amp.autocast():
                    output = self.model(input_tensor)
            self.stream.synchronize()
        return output
