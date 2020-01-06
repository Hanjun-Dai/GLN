import torch
import extlib
try:
    import extlib_cuda
except:
    print('not loading cuda jagged ops')
from torch.autograd import Function
from torch.nn import Module
import numpy as np

#----------------------
#   jagged_log_softmax
#----------------------
class JaggedLogSoftmaxFunc(Function):
    @staticmethod
    def forward(ctx, logits, prefix_sum):
        assert len(prefix_sum.size()) == 1        
        if not logits.is_cuda:
            output = extlib.jagged_log_softmax_forward(logits, prefix_sum)
        else:
            output = extlib_cuda.jagged_log_softmax_forward_cuda(logits, prefix_sum)

        ctx.save_for_backward(prefix_sum, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        prefix_sum, output = ctx.saved_variables
        if not grad_output.is_cuda:
            grad_input = extlib.jagged_log_softmax_backward(output.data, grad_output, prefix_sum.data)
        else:
            grad_input = extlib_cuda.jagged_log_softmax_backward_cuda(output.data, grad_output, prefix_sum.data)        
        return grad_input, None


class JaggedLogSoftmax(Module):
    def forward(self, logits, prefix_sum):
        return JaggedLogSoftmaxFunc.apply(logits, prefix_sum)

jagged_log_softmax = JaggedLogSoftmax()

