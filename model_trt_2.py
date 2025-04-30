import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from transforms import test_transform_fn
import cv2

class StereoRT:
    def __init__(self, engine_path, device='cuda'):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.transform = test_transform_fn()
        self.device = device

        # Get input/output tensor names
        tensor_names = self._get_tensor_names()
        self.input_name = tensor_names['input']
        self.output_name = tensor_names['output']

        # Now you can safely get their shapes
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # Allocate host and device memory
        self.h_input = cuda.pagelocked_empty(shape=self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(shape=self.output_shape, dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Binding addresses list
        self.bindings = [None] * self.engine.num_io_tensors
        self.bindings[self.engine.get_tensor_location(self.input_name)] = int(self.d_input)
        self.bindings[self.engine.get_tensor_location(self.output_name)] = int(self.d_output)

        # CUDA stream
        self.stream = cuda.Stream()

    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_tensor_names(self):
        names = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                names['input'] = name
            elif mode == trt.TensorIOMode.OUTPUT:
                names['output'] = name
        if 'input' not in names or 'output' not in names:
            raise RuntimeError("Could not find valid input/output tensor names.")
        return names


    def _prepare_input(self, left_image, right_image):
        left_tensor = self.transform(left_image).unsqueeze(0).numpy()
        right_tensor = self.transform(right_image).unsqueeze(0).numpy()
        input_tensor = np.concatenate((left_tensor, right_tensor), axis=1).astype(np.float32)
        return input_tensor

    def inference(self, left_image, right_image):
        preprocess_start = time.time()
        input_tensor = self._prepare_input(left_image, right_image)
        preprocess_end = time.time()
        print(f"Preprocessing Time: {(preprocess_end - preprocess_start)*100:.3f} ms")

        np.copyto(self.h_input, input_tensor)

        # For dynamic shapes: must set shape explicitly first!
        self.context.set_input_shape(self.input_name, self.input_shape)

        # Then set memory bindings
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        # Timing
        start_event = cuda.Event()
        end_event = cuda.Event()
        start_event.record(stream=self.stream)

        # Run async inference
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)

        end_event.record(stream=self.stream)
        self.stream.synchronize()

        print(f"Inference Time (GPU): {start_event.time_till(end_event):.3f} ms")
        return self.h_output

