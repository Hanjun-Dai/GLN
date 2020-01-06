#include "extlib_cuda.h"
#include "extlib_cuda_kernels.h"
#include <cfloat>
#include <cstdint>

torch::Tensor make_contiguous(torch::Tensor& t)
{
    if (t.is_contiguous())
        return t;
    return t.contiguous();
}

torch::Tensor jagged_log_softmax_forward_cuda(torch::Tensor logits, torch::Tensor prefix_sum)
{
    logits = make_contiguous(logits);
    prefix_sum = make_contiguous(prefix_sum);    
    auto output = torch::zeros_like(logits);
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t* ps = prefix_sum.data<int64_t>();

    AT_DISPATCH_FLOATING_TYPES(logits.type(), "jagged_log_softmax_forward_cuda", ([&] {        
        HostLogSoftmaxForward(logits.data<scalar_t>(),
                              output.data<scalar_t>(),                               
                              ps, bsize);
    }));
    return output;
}

torch::Tensor jagged_log_softmax_backward_cuda(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum)
{
    output = make_contiguous(output);
    grad_output = make_contiguous(grad_output);
    prefix_sum = make_contiguous(prefix_sum);

    auto grad_input = torch::zeros_like(output);

    int64_t bsize = prefix_sum.sizes()[0];
    int64_t* ps = prefix_sum.data<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(output.type(), "jagged_log_softmax_backward_cuda", ([&] {        
        HostLogSoftmaxBackward(grad_output.data<scalar_t>(),
                               grad_input.data<scalar_t>(),
                               output.data<scalar_t>(),
                               ps, bsize);
    }));
    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("jagged_log_softmax_forward_cuda", &jagged_log_softmax_forward_cuda, "Jagged Log Softmax Forward (CUDA)");
    m.def("jagged_log_softmax_backward_cuda", &jagged_log_softmax_backward_cuda, "Jagged Log Softmax Backward (CUDA)");    
}