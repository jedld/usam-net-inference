import torch
from transforms import test_transform_fn

class StereoRT:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.transform = test_transform_fn()  # Include transform here
        self.stream = torch.cuda.Stream()     # Custom stream instance

    def _load_model(self, path):
        model = torch.jit.load(path, map_location=self.device)
        return model.eval()

    def _prepare_input(self, left_image, right_image):
        with torch.cuda.stream(self.stream):
            left_tensor = self.transform(left_image).unsqueeze(0).to(self.device, non_blocking=True)
            right_tensor = self.transform(right_image).unsqueeze(0).to(self.device, non_blocking=True)
            input_tensor = torch.cat((left_tensor, right_tensor), dim=1)

        self.stream.synchronize()  # Ensure transform is complete before inference
        return input_tensor

    def inference(self, left_image, right_image):
        with torch.no_grad():
            input_tensor = self._prepare_input(left_image, right_image)
            with torch.cuda.stream(self.stream):
                output = self.model(input_tensor)
        return output
