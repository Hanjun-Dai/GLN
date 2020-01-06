#include "extlib.h"
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <algorithm>

torch::Tensor make_contiguous(torch::Tensor& t)
{
    if (t.is_contiguous())
        return t;
    return t.contiguous();
}

template<typename scalar_t>
void impl_jagged_log_softmax_forward(scalar_t *input_data_base, scalar_t *output_data_base, torch::Tensor prefix_sum)
{
    int64_t *ps = prefix_sum.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t i, d;
    
    #pragma omp parallel for private(i, d)
    for (i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        
        scalar_t* input_data  = input_data_base  + offset;
        scalar_t* output_data = output_data_base + offset;
        
        int64_t n_ele = ps[i] - offset;
        scalar_t max_input = -FLT_MAX;
        
        for (d = 0; d < n_ele; d++)
            max_input = std::max(max_input, input_data[d]);
            
        double logsum = 0;
        for (d = 0; d < n_ele; d++)
            logsum += exp(input_data[d] - max_input);
        logsum = max_input + log(logsum);
        for (d = 0; d < n_ele; d++)
            output_data[d] = input_data[d] - logsum;
    }
}


template void impl_jagged_log_softmax_forward<float>(float *input_data_base, float *output_data_base, torch::Tensor prefix_sum);
template void impl_jagged_log_softmax_forward<double>(double *input_data_base, double *output_data_base, torch::Tensor prefix_sum);


torch::Tensor jagged_log_softmax_forward(torch::Tensor logits, torch::Tensor prefix_sum)
{
    logits = make_contiguous(logits);
    prefix_sum = make_contiguous(prefix_sum);
    auto output = torch::zeros_like(logits);
    AT_DISPATCH_FLOATING_TYPES(logits.type(), "jagged_log_softmax_forward", ([&] {
        impl_jagged_log_softmax_forward(logits.data<scalar_t>(), 
                                        output.data<scalar_t>(),                                         
                                        prefix_sum);
    }));
    return output;
}

template<typename scalar_t>
void impl_jagged_log_softmax_backward(scalar_t *output_data_base, scalar_t *gradOutput_data_base, torch::Tensor prefix_sum, scalar_t *gradInput_data_base)
{
    int64_t *ps = prefix_sum.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t i, d;

    #pragma omp parallel for private(i, d)
    for (i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        scalar_t *gradInput_data  = gradInput_data_base  + offset;
        scalar_t *output_data     = output_data_base     + offset;
        scalar_t *gradOutput_data = gradOutput_data_base + offset;
        
        double sum = 0;
        int64_t n_ele = ps[i] - offset;
        for (d = 0; d < n_ele; d++)
            sum += gradOutput_data[d];
        
        for (d = 0; d < n_ele; d++)
            gradInput_data[d] = gradOutput_data[d] - exp(output_data[d]) * sum;
    }
}

template void impl_jagged_log_softmax_backward<float>(float *output_data_base, float *gradOutput_data_base, torch::Tensor prefix_sum, float *gradInput_data_base);
template void impl_jagged_log_softmax_backward<double>(double *output_data_base, double *gradOutput_data_base, torch::Tensor prefix_sum, double *gradInput_data_base);

torch::Tensor jagged_log_softmax_backward(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum)
{
    output = make_contiguous(output);
    grad_output = make_contiguous(grad_output);
    prefix_sum = make_contiguous(prefix_sum);

    auto grad_input = torch::zeros_like(output);

    AT_DISPATCH_FLOATING_TYPES(output.type(), "jagged_log_softmax_backward", ([&] {
        impl_jagged_log_softmax_backward(output.data<scalar_t>(), 
                                         grad_output.data<scalar_t>(), 
                                         prefix_sum,
                                         grad_input.data<scalar_t>());
    }));

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jagged_log_softmax_forward", &jagged_log_softmax_forward, "Jagged Log Softmax Forward");
  m.def("jagged_log_softmax_backward", &jagged_log_softmax_backward, "Jagged Log Softmax Backward");  
}
