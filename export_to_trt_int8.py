import sys
sys.path.append('..')
from model import SAStereoCNN2
import torch
import os
import torch_tensorrt
from torch.utils.data import DataLoader
from post_training_dataset import CalibrationStereoDataset
from transforms import test_transform_fn
from torch_tensorrt.ts.ptq import DataLoaderCalibrator

CHECKPOINT_PATH = '../stereo_cnn_stereo_cnn_sa_baseline.checkpoint'
data_set_folder = 'data'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SAStereoCNN2(device)
    model.to(device)
    if os.path.exists(CHECKPOINT_PATH):
        print("loading existing checkpoint ...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
        model.load_state_dict(checkpoint)

    model.eval()
    
    dummy_input = torch.randn(1, 6, 400, 879).to(device)
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save("stereo_cnn.ts")

    # Load scripted model
    scripted_model = torch.jit.load("stereo_cnn.ts").eval().cuda()

    calibration_transform = test_transform_fn()

    calibration_dataset = CalibrationStereoDataset(
        left_dir=f'{data_set_folder}/train/left_images',
        right_dir=f'{data_set_folder}/train/right_images',
        transform=calibration_transform
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Compile with INT8 precision
    trt_model = torch_tensorrt.compile(
        scripted_model,
        ir="torchscript",
        inputs=[torch_tensorrt.Input((1, 6, 400, 879), dtype=torch.float32)],
        enabled_precisions={torch.int8},
        calibrator=DataLoaderCalibrator(calibration_loader, cache_file="int8_calib.cache"),
        require_full_compilation=True,
        truncate_long_and_double=True
    )

    trt_model.save("model_trt_int8.ts")
    print("INT8 model saved as model_trt_int8.ts")

if __name__ == "__main__":
    main()
