#include "stereo_cnn_cuda.h"
#include <torch/script.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <filesystem>

namespace py = pybind11;

StereoModelCUDA::StereoModelCUDA() {
    // Initialize all the layers with the correct parameters
    // Downsampling blocks
    down1_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 64, 3).stride(2).padding(1));
    down1_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    
    down2_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1));
    down2_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
    
    down3_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1));
    down3_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
    
    down4_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1));
    down4_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
    
    down5_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1));
    down5_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(1024));
    
    // Self-attention
    sa_query_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024 / 8, 1));
    sa_key_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024 / 8, 1));
    sa_value_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024, 1));
    
    // Upsampling blocks
    up1_conv = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(1024, 512, 3).stride(2).padding(1));
    up1_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(512));
    
    up2_conv = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 256, 4).stride(2).padding(1));
    up2_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256));
    
    up3_conv = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1));
    up3_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128));
    
    up4_conv = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1));
    up4_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    
    up5_conv = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, {4, 3}).stride(2).padding(1));
    up5_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    
    // Final convolution blocks
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1));
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1));
    conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 1).stride(1).padding(0));
    
    // Register all modules
    register_module("down1_conv", down1_conv);
    register_module("down1_bn", down1_bn);
    
    register_module("down2_conv", down2_conv);
    register_module("down2_bn", down2_bn);
    
    register_module("down3_conv", down3_conv);
    register_module("down3_bn", down3_bn);
    
    register_module("down4_conv", down4_conv);
    register_module("down4_bn", down4_bn);
    
    register_module("down5_conv", down5_conv);
    register_module("down5_bn", down5_bn);
    
    register_module("sa_query_conv", sa_query_conv);
    register_module("sa_key_conv", sa_key_conv);
    register_module("sa_value_conv", sa_value_conv);
    
    register_module("up1_conv", up1_conv);
    register_module("up1_bn", up1_bn);
    
    register_module("up2_conv", up2_conv);
    register_module("up2_bn", up2_bn);
    
    register_module("up3_conv", up3_conv);
    register_module("up3_bn", up3_bn);
    
    register_module("up4_conv", up4_conv);
    register_module("up4_bn", up4_bn);
    
    register_module("up5_conv", up5_conv);
    register_module("up5_bn", up5_bn);
    
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    
    // Move all parameters to the GPU
    this->to(device);
}

StereoModelCUDA::~StereoModelCUDA() {
    // Destructor
}

torch::Tensor StereoModelCUDA::forward(torch::Tensor x) {
    // If we have a TorchScript module, use it directly
    if (using_torch_module) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x.to(device));
        
        try {
            torch::Tensor output = torch_module.forward(inputs).toTensor();
            return output;
        } catch (const c10::Error& e) {
            std::cerr << "Error during forward pass with TorchScript module: " << e.what() << std::endl;
            throw;
        }
    }
    
    // Otherwise use our implemented network
    // Ensure input is on the correct device
    x = x.to(device);
    
    // Downsampling path
    auto down1 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            conv2d_forward_cuda(x, down1_conv->weight, down1_conv->bias, 2, 1),
            down1_bn->weight, down1_bn->bias, 
            down1_bn->running_mean, down1_bn->running_var,
            1e-5
        ),
        0.01
    );
    
    auto down2 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            conv2d_forward_cuda(down1, down2_conv->weight, down2_conv->bias, 2, 1),
            down2_bn->weight, down2_bn->bias, 
            down2_bn->running_mean, down2_bn->running_var,
            1e-5
        ),
        0.01
    );
    
    auto down3 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            conv2d_forward_cuda(down2, down3_conv->weight, down3_conv->bias, 2, 1),
            down3_bn->weight, down3_bn->bias, 
            down3_bn->running_mean, down3_bn->running_var,
            1e-5
        ),
        0.01
    );
    
    auto down4 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            conv2d_forward_cuda(down3, down4_conv->weight, down4_conv->bias, 2, 1),
            down4_bn->weight, down4_bn->bias, 
            down4_bn->running_mean, down4_bn->running_var,
            1e-5
        ),
        0.01
    );
    
    auto down5 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            conv2d_forward_cuda(down4, down5_conv->weight, down5_conv->bias, 2, 1),
            down5_bn->weight, down5_bn->bias, 
            down5_bn->running_mean, down5_bn->running_var,
            1e-5
        ),
        0.01
    );
    
    // Self-attention
    auto sa = self_attention_forward_cuda(
        down5, 
        sa_query_conv->weight, sa_query_conv->bias,
        sa_key_conv->weight, sa_key_conv->bias,
        sa_value_conv->weight, sa_value_conv->bias
    );
    
    // Upsampling path with skip connections
    auto up1 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            convtranspose2d_forward_cuda(sa, up1_conv->weight, up1_conv->bias, 2, 1),
            up1_bn->weight, up1_bn->bias, 
            up1_bn->running_mean, up1_bn->running_var,
            1e-5
        ),
        0.01
    ) + down4;
    
    auto up2 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            convtranspose2d_forward_cuda(up1, up2_conv->weight, up2_conv->bias, 2, 1),
            up2_bn->weight, up2_bn->bias, 
            up2_bn->running_mean, up2_bn->running_var,
            1e-5
        ),
        0.01
    ) + down3;
    
    auto up3 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            convtranspose2d_forward_cuda(up2, up3_conv->weight, up3_conv->bias, 2, 1),
            up3_bn->weight, up3_bn->bias, 
            up3_bn->running_mean, up3_bn->running_var,
            1e-5
        ),
        0.01
    ) + down2;
    
    auto up4 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            convtranspose2d_forward_cuda(up3, up4_conv->weight, up4_conv->bias, 2, 1),
            up4_bn->weight, up4_bn->bias, 
            up4_bn->running_mean, up4_bn->running_var,
            1e-5
        ),
        0.01
    ) + down1;
    
    auto up5 = leaky_relu_forward_cuda(
        batch_norm_forward_cuda(
            convtranspose2d_forward_cuda(up4, up5_conv->weight, up5_conv->bias, 2, 1),
            up5_bn->weight, up5_bn->bias, 
            up5_bn->running_mean, up5_bn->running_var,
            1e-5
        ),
        0.01
    );
    
    // Final convolution layers
    auto out = leaky_relu_forward_cuda(
        conv2d_forward_cuda(up5, conv1->weight, conv1->bias, 1, 1),
        0.01
    );
    
    out = leaky_relu_forward_cuda(
        conv2d_forward_cuda(out, conv2->weight, conv2->bias, 1, 1),
        0.01
    );
    
    out = sigmoid_forward_cuda(
        conv2d_forward_cuda(out, conv3->weight, conv3->bias, 1, 0)
    );
    
    // Multiply by 255 as in original model
    return out * 255.0;
}

void StereoModelCUDA::load_weights(const std::string& checkpoint_path) {
    try {
        // First, check if the file exists
        if (!std::filesystem::exists(checkpoint_path)) {
            std::cerr << "Error: Checkpoint file does not exist at " << checkpoint_path << std::endl;
            throw std::runtime_error("Checkpoint file not found");
        }
        
        // Load the model directly as a TorchScript module
        std::cout << "Loading TorchScript model: " << checkpoint_path << std::endl;
        
        try {
            // Load TorchScript module
            torch::jit::script::Module module = torch::jit::load(checkpoint_path);
            std::cout << "Loaded TorchScript module successfully" << std::endl;
            
            // Try to handle this module directly - with the assumption
            // that we can call the model's forward method directly from C++
            torch_module = module;
            using_torch_module = true;
            
            std::cout << "Successfully loaded model from " << checkpoint_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading TorchScript model: " << e.what() << std::endl;
            std::cerr << "Make sure you're using a proper TorchScript model (scripted or traced)" << std::endl;
            std::cerr << "You can create one using: python save_scripted_model.py" << std::endl;
            throw;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, torch::Tensor> StereoModelCUDA::state_dict() {
    std::unordered_map<std::string, torch::Tensor> result;
    auto params = this->named_parameters();
    
    for (auto& param : params) {
        result[param.key()] = param.value();
    }
    
    return result;
}

void StereoModelCUDA::load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state_dict) {
    // Get all our parameters
    auto params = this->named_parameters();
    
    // Copy values from state_dict to our parameters
    for (const auto& pair : state_dict) {
        const std::string& name = pair.first;
        const torch::Tensor& value = pair.second;
        
        // Find parameter by name
        for (auto& named_param : params) {
            if (named_param.key() == name) {
                named_param.value().copy_(value);
                break;
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Explicitly specify std::shared_ptr as holder
    py::class_<StereoModelCUDA, std::shared_ptr<StereoModelCUDA>>(m, "StereoModelCUDA")
        .def(py::init([]() { return std::make_shared<StereoModelCUDA>(); }))
        .def("forward", &StereoModelCUDA::forward)
        .def("load_weights", &StereoModelCUDA::load_weights)
        .def("state_dict", &StereoModelCUDA::state_dict)
        .def("load_state_dict", &StereoModelCUDA::load_state_dict);

    // Expose CUDA functions
    m.def("conv2d_forward_cuda", &conv2d_forward_cuda, "Conv2d forward (CUDA)");
    m.def("convtranspose2d_forward_cuda", &convtranspose2d_forward_cuda, "ConvTranspose2d forward (CUDA)");
    m.def("leaky_relu_forward_cuda", &leaky_relu_forward_cuda, "LeakyReLU forward (CUDA)");
    m.def("batch_norm_forward_cuda", &batch_norm_forward_cuda, "BatchNorm2d forward (CUDA)");
    m.def("self_attention_forward_cuda", &self_attention_forward_cuda, "Self-attention forward (CUDA)");
    m.def("sigmoid_forward_cuda", &sigmoid_forward_cuda, "Sigmoid forward (CUDA)");
}