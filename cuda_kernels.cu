#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_kernels.h"

// CUDA kernels for forward operations

// ================ UTILITY MACROS ================
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// GPU kernel for 2D convolution
__global__ void conv2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_h, int kernel_w, int stride, int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_channels * out_height * out_width) return;
    
    // Compute indices for output dimensions
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    // Initialize output value with bias
    float sum = bias ? bias[oc] : 0.0f;
    
    // Compute convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Write result to output
    output[idx] = sum;
}

torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                               int stride, int padding) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_w) / stride + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::device(input.device()).dtype(input.dtype()));
    
    // Set grid and block dimensions for CUDA kernel
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;
    
    // Launch kernel
    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_h, kernel_w, stride, padding
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

// GPU kernel for 2D transpose convolution (simplified)
__global__ void convtranspose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int out_height, int out_width,
    int kernel_h, int kernel_w, int stride, int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_channels * out_height * out_width) return;
    
    // Compute indices for output dimensions
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    // Initialize output value with bias
    float sum = bias ? bias[oc] : 0.0f;
    
    // Compute transposed convolution (simplified)
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = (oh + padding - kh) / stride;
                int iw = (ow + padding - kw) / stride;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width &&
                    (oh + padding - kh) % stride == 0 && (ow + padding - kw) % stride == 0) {
                    
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Write result to output
    output[idx] = sum;
}

torch::Tensor convtranspose2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                        int stride, int padding) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int out_height = (in_height - 1) * stride + kernel_h - 2 * padding;
    int out_width = (in_width - 1) * stride + kernel_w - 2 * padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               torch::device(input.device()).dtype(input.dtype()));
    
    // Set grid and block dimensions for CUDA kernel
    const int threads = 256;
    const int blocks = (batch_size * out_channels * out_height * out_width + threads - 1) / threads;
    
    // Launch kernel
    convtranspose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, out_height, out_width,
        kernel_h, kernel_w, stride, padding
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    return output;
}

// GPU kernel for LeakyReLU activation function
__global__ void leaky_relu_kernel(
    const float* input, float* output, int size, float negative_slope) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = input[idx];
        output[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_cuda(torch::Tensor input, float negative_slope) {
    CHECK_INPUT(input);
    
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    // Set grid and block dimensions for CUDA kernel
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Launch kernel
    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size, negative_slope
    );
    
    return output;
}

// GPU kernel for BatchNorm2d
__global__ void batch_norm_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    float* output, int size, int features, int spatial_size, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        int feature_idx = (idx / spatial_size) % features;
        
        float mean = running_mean[feature_idx];
        float var = running_var[feature_idx];
        float gamma = weight ? weight[feature_idx] : 1.0f;
        float beta = bias ? bias[feature_idx] : 0.0f;
        
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        output[idx] = gamma * normalized + beta;
    }
}

torch::Tensor batch_norm_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                  torch::Tensor running_mean, torch::Tensor running_var, 
                                  float eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(running_mean);
    CHECK_INPUT(running_var);
    if (weight.defined()) CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int features = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int spatial_size = height * width;
    int size = batch_size * features * spatial_size;
    
    // Set grid and block dimensions for CUDA kernel
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Launch kernel
    batch_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(), 
        running_var.data_ptr<float>(),
        output.data_ptr<float>(), size, features, spatial_size, eps
    );
    
    return output;
}

// GPU kernel for self-attention
__global__ void self_attention_kernel(
    const float* input, const float* query_weight, const float* query_bias,
    const float* key_weight, const float* key_bias,
    const float* value_weight, const float* value_bias,
    float* output, int batch_size, int channels, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx >= total_size) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);
    
    // Compute query, key, value for current position
    int reduced_channels = channels / 8;
    float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
    
    // Compute query value for current position
    for (int ic = 0; ic < channels; ++ic) {
        int input_idx = ((b * channels + ic) * height + h) * width + w;
        int weight_idx = c * channels + ic;
        q_val += input[input_idx] * query_weight[weight_idx];
    }
    if (query_bias) q_val += query_bias[c];
    
    // Compute key value for current position
    for (int ic = 0; ic < channels; ++ic) {
        int input_idx = ((b * channels + ic) * height + h) * width + w;
        int weight_idx = c * channels + ic;
        k_val += input[input_idx] * key_weight[weight_idx];
    }
    if (key_bias) k_val += key_bias[c];
    
    // Compute value for current position
    for (int ic = 0; ic < channels; ++ic) {
        int input_idx = ((b * channels + ic) * height + h) * width + w;
        int weight_idx = c * channels + ic;
        v_val += input[input_idx] * value_weight[weight_idx];
    }
    if (value_bias) v_val += value_bias[c];
    
    // Compute attention
    float attention = 0.0f;
    float sum_exp = 0.0f;
    
    // Simplified attention calculation (note: in a real implementation this would need to be more sophisticated)
    for (int i = 0; i < spatial_size; ++i) {
        float a = expf(q_val * k_val);
        sum_exp += a;
        attention += a * v_val;
    }
    
    attention /= sum_exp;
    
    // Add skip connection
    output[idx] = attention + input[idx];
}

torch::Tensor self_attention_forward_cuda(torch::Tensor input, torch::Tensor query_weight, torch::Tensor query_bias,
                                      torch::Tensor key_weight, torch::Tensor key_bias,
                                      torch::Tensor value_weight, torch::Tensor value_bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(query_weight);
    CHECK_INPUT(key_weight);
    CHECK_INPUT(value_weight);
    if (query_bias.defined()) CHECK_INPUT(query_bias);
    if (key_bias.defined()) CHECK_INPUT(key_bias);
    if (value_bias.defined()) CHECK_INPUT(value_bias);
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    auto output = torch::empty_like(input);
    
    // Set grid and block dimensions for CUDA kernel
    const int threads = 256;
    const int blocks = (batch_size * channels * height * width + threads - 1) / threads;
    
    // Launch kernel
    self_attention_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        query_weight.data_ptr<float>(), 
        query_bias.defined() ? query_bias.data_ptr<float>() : nullptr,
        key_weight.data_ptr<float>(), 
        key_bias.defined() ? key_bias.data_ptr<float>() : nullptr,
        value_weight.data_ptr<float>(), 
        value_bias.defined() ? value_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), batch_size, channels, height, width
    );
    
    return output;
}

// GPU kernel for sigmoid activation function
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

torch::Tensor sigmoid_forward_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    // Set grid and block dimensions for CUDA kernel
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    // Launch kernel
    sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size
    );
    
    return output;
} 