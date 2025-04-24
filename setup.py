from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="stereo_cnn_cuda",
    version="0.1",
    author="CUDA Stereo CNN",
    author_email="",
    description="CUDA implementation of the Stereo CNN model",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="stereo_cnn_cuda",
            sources=[
                "stereo_cnn_cuda.cpp",
                "cuda_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    install_requires=[
        "torch>=1.7.0",
    ],
) 