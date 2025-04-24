#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cuda_kernels.h"

namespace py = pybind11;

class StereoModelCUDA : public torch::nn::Module {
public:
    StereoModelCUDA();
    ~StereoModelCUDA();
    
    void load_weights(const std::string& checkpoint_path);
    torch::Tensor forward(torch::Tensor x);
    
    // Pytorch state_dict compatible method
    std::unordered_map<std::string, torch::Tensor> state_dict();
    void load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict);

private:
    // TorchScript module to use instead of our manual implementation
    torch::jit::script::Module torch_module;
    bool using_torch_module = false;
    
    // Down blocks
    torch::nn::Conv2d down1_conv{nullptr};
    torch::nn::BatchNorm2d down1_bn{nullptr};
    
    torch::nn::Conv2d down2_conv{nullptr};
    torch::nn::BatchNorm2d down2_bn{nullptr};
    
    torch::nn::Conv2d down3_conv{nullptr};
    torch::nn::BatchNorm2d down3_bn{nullptr};
    
    torch::nn::Conv2d down4_conv{nullptr};
    torch::nn::BatchNorm2d down4_bn{nullptr};
    
    torch::nn::Conv2d down5_conv{nullptr};
    torch::nn::BatchNorm2d down5_bn{nullptr};
    
    // Self-attention
    torch::nn::Conv2d sa_query_conv{nullptr};
    torch::nn::Conv2d sa_key_conv{nullptr};
    torch::nn::Conv2d sa_value_conv{nullptr};
    
    // Up blocks
    torch::nn::ConvTranspose2d up1_conv{nullptr};
    torch::nn::BatchNorm2d up1_bn{nullptr};
    
    torch::nn::ConvTranspose2d up2_conv{nullptr};
    torch::nn::BatchNorm2d up2_bn{nullptr};
    
    torch::nn::ConvTranspose2d up3_conv{nullptr};
    torch::nn::BatchNorm2d up3_bn{nullptr};
    
    torch::nn::ConvTranspose2d up4_conv{nullptr};
    torch::nn::BatchNorm2d up4_bn{nullptr};
    
    torch::nn::ConvTranspose2d up5_conv{nullptr};
    torch::nn::BatchNorm2d up5_bn{nullptr};
    
    // Final conv layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};
    
    // Use GPU for computation
    torch::Device device{torch::kCUDA};
}; 