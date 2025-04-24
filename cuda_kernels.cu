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

// ================ UTILITY FUNCTIONS ================
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

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

// Block size for flash attention
#define BLOCK_SIZE_M 32  // Number of rows
#define BLOCK_SIZE_N 32  // Number of columns
#define BLOCK_SIZE_K 32  // Internal block size

// Flash attention kernel - computes attention more efficiently with tiling
__global__ void flash_attention_kernel(
    const float* query,    // [B, H, W, C]
    const float* key,      // [B, H, W, C]
    const float* value,    // [B, H, W, C]
    float* output,         // [B, H, W, C]
    const float* input,    // Original input for residual connection [B, C, H, W]
    int batch_size, int height, int width, int channels,
    float scaling_factor) {
    
    // Calculate spatial size and configure shared memory
    const int spatial_size = height * width;
    const int num_heads = 1; // For simplicity, using single head
    const int head_dim = channels;
    
    // Shared memory for blocks of Q, K, V
    __shared__ float s_query[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float s_key[BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ float s_value[BLOCK_SIZE_N][BLOCK_SIZE_K];
    
    // Each thread block handles a block of the output matrix
    const int block_row = blockIdx.x * BLOCK_SIZE_M;
    const int block_col = blockIdx.y * BLOCK_SIZE_N;
    
    // Thread indices
    const int thread_row = threadIdx.x;
    const int thread_col = threadIdx.y;
    
    // Batch index
    const int b = blockIdx.z;
    
    // Accumulator for output
    float acc[BLOCK_SIZE_M] = {0.0f};
    
    // Track maximum value for numerical stability
    float m_i[BLOCK_SIZE_M] = {-INFINITY};
    float l_i[BLOCK_SIZE_M] = {0.0f};
    
    // Process blocks of K dimension
    for (int bk = 0; bk < (spatial_size + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++bk) {
        // Load Q, K, V blocks into shared memory
        const int k_offset = bk * BLOCK_SIZE_K;
        
        // Load Q block
        if (block_row + thread_row < spatial_size && k_offset + thread_col < head_dim) {
            int q_idx = b * spatial_size * head_dim + (block_row + thread_row) * head_dim + (k_offset + thread_col);
            s_query[thread_row][thread_col] = query[q_idx];
        } else {
            s_query[thread_row][thread_col] = 0.0f;
        }
        
        // Load K block (transposed)
        if (k_offset + thread_row < spatial_size && block_col + thread_col < head_dim) {
            int k_idx = b * spatial_size * head_dim + (k_offset + thread_row) * head_dim + (block_col + thread_col);
            s_key[thread_row][thread_col] = key[k_idx];
        } else {
            s_key[thread_row][thread_col] = 0.0f;
        }
        
        // Load V block
        if (k_offset + thread_row < spatial_size && block_col + thread_col < head_dim) {
            int v_idx = b * spatial_size * head_dim + (k_offset + thread_row) * head_dim + (block_col + thread_col);
            s_value[thread_row][thread_col] = value[v_idx];
        } else {
            s_value[thread_row][thread_col] = 0.0f;
        }
        
        __syncthreads();
        
        // Calculate attention scores for this block
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            // Safe to calculate only if we're within bounds
            if (block_row + thread_row < spatial_size && k_offset + k < spatial_size) {
                float qk = s_query[thread_row][k] * s_key[k][thread_col] * scaling_factor;
                
                // Numerically stable update for softmax
                float m_prev = m_i[thread_row];
                float m_new = fmaxf(m_prev, qk);
                
                float exp_new = expf(qk - m_new);
                float exp_old = l_i[thread_row] * expf(m_prev - m_new);
                
                // Update running softmax
                l_i[thread_row] = exp_old + exp_new;
                m_i[thread_row] = m_new;
                
                // Update accumulator
                acc[thread_row] = (acc[thread_row] * exp_old + s_value[k][thread_col] * exp_new) / l_i[thread_row];
            }
        }
        
        __syncthreads();
    }
    
    // Write back results
    if (block_row + thread_row < spatial_size && block_col + thread_col < head_dim) {
        int out_idx = b * spatial_size * head_dim + (block_row + thread_row) * head_dim + (block_col + thread_col);
        
        // Convert back to BCHW format and add residual connection
        int input_idx = (b * channels + block_col + thread_col) * height * width + (block_row + thread_row);
        output[out_idx] = acc[thread_row] + input[input_idx];
    }
}

// Helper function to compute QKV projections
torch::Tensor compute_qkv_projections(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Reshape for projection: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto spatial_size = height * width;
    
    auto reshaped = input.reshape({batch_size, channels, spatial_size}).permute({0, 2, 1});
    
    // Apply linear projection
    auto output = torch::matmul(reshaped, weight.t());
    
    if (bias.defined()) {
        output = output + bias.unsqueeze(0).unsqueeze(0);
    }
    
    // Return in [B, H*W, C] format for attention calculation
    return output;
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
    int spatial_size = height * width;
    
    // Compute Q, K, V projections efficiently using matrix multiplication
    auto query = compute_qkv_projections(input, query_weight, query_bias);
    auto key = compute_qkv_projections(input, key_weight, key_bias);
    auto value = compute_qkv_projections(input, value_weight, value_bias);
    
    // Prepare output tensor - use same shape as input
    auto output = torch::empty_like(query);
    
    // Set scaling factor for attention scores (prevents softmax saturation)
    float scaling_factor = 1.0f / sqrtf(channels);
    
    // Configure grid and blocks for flash attention
    dim3 threads(BLOCK_SIZE_M, BLOCK_SIZE_N);
    dim3 grid(
        (spatial_size + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        (channels + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        batch_size
    );
    
    // Launch flash attention kernel
    flash_attention_kernel<<<grid, threads>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch_size, height, width, channels,
        scaling_factor
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in flash attention: %s\n", cudaGetErrorString(err));
    }
    
    // Reshape output back to BCHW format
    auto reshaped_output = output.permute({0, 2, 1}).reshape({batch_size, channels, height, width});
    
    return reshaped_output;
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