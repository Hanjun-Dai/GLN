#ifndef EXTLIB_CUDA_H
#define EXTLIB_CUDA_H

#include <torch/extension.h>

torch::Tensor jagged_log_softmax_forward_cuda(torch::Tensor logits, torch::Tensor prefix_sum);

torch::Tensor jagged_log_softmax_backward_cuda(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum);


#endif