#ifndef EXTLIB_H
#define EXTLIB_H

#include <torch/extension.h>
#include <vector>

torch::Tensor jagged_log_softmax_forward(torch::Tensor logits, torch::Tensor prefix_sum);

torch::Tensor jagged_log_softmax_backward(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum);


#endif
