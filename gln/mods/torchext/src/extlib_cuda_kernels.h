#ifndef EXTLIB_CUDA_KERNELS_H
#define EXTLIB_CUDA_KERNELS_H

#include <cstdint>

template<typename scalar_t>
void HostLogSoftmaxForward(scalar_t* input, scalar_t *output, int64_t* ps, int64_t bsize);

template<typename scalar_t>
void HostLogSoftmaxBackward(scalar_t *gradOutput, scalar_t *gradInput, scalar_t *output, int64_t* ps, int64_t bsize);


#endif