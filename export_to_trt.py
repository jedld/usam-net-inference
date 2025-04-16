import sys
sys.path.append('..')
from model import SAStereoCNN2
import torch
import os
import torch_tensorrt

CHECKPOINT_PATH = '../stereo_cnn_stereo_cnn_sa_baseline.checkpoint'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SAStereoCNN2(device)
    model.to(device)
    if os.path.exists(CHECKPOINT_PATH):
        print("loading existing checkpoint ...")
        checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
        model.load_state_dict(checkpoint)

    # Prepare dummy input
    dummy_left = torch.randn(1, 3, *(400, 879)).to('cuda')
    dummy_right = torch.randn(1, 3, *(400, 879)).to('cuda')
    dummy_input = torch.cat((dummy_left, dummy_right), dim=1)

    # Your PyTorch model (must be in eval mode)
    model.eval()

    # Script the model
    scripted_model = torch.jit.trace(model, dummy_input)  # or torch.jit.script(model)

    # Save the scripted model
    scripted_model.save("stereo_cnn.ts")

    scripted_model = torch.jit.load("stereo_cnn.ts").eval().cuda()

    trt_model = torch_tensorrt.compile(
    scripted_model,
    ir="torchscript",
    inputs=[torch_tensorrt.Input((1, 6, 400, 879), dtype=torch.float16)],
    enabled_precisions={torch.float16},
    require_full_compilation=True,
    truncate_long_and_double=True   # 👈 This converts int64/float64 to int32/float32
    )

    # Save optimized TRT model (optional)
    trt_model.save("model_trt.ts")