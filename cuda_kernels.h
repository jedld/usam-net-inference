#pragma once

#include <torch/extension.h>

// CUDA forward declarations
torch::Tensor conv2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                               int stride, int padding);
torch::Tensor convtranspose2d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                        int stride, int padding);
torch::Tensor leaky_relu_forward_cuda(torch::Tensor input, float negative_slope);
torch::Tensor batch_norm_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                  torch::Tensor running_mean, torch::Tensor running_var, 
                                  float eps = 1e-5);
torch::Tensor self_attention_forward_cuda(torch::Tensor input, torch::Tensor query_weight, torch::Tensor query_bias,
                                      torch::Tensor key_weight, torch::Tensor key_bias,
                                      torch::Tensor value_weight, torch::Tensor value_bias);
torch::Tensor sigmoid_forward_cuda(torch::Tensor input);
