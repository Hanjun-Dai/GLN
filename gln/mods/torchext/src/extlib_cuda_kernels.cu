#include "extlib_cuda_kernels.h"
#include <stdexcept>
#include <float.h>

struct SharedMem
{
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};


struct Max
{
    template<typename scalar_t>
    __device__ __forceinline__ double operator()(double x, scalar_t y) const {
        return x > static_cast<double>(y) ? x : static_cast<double>(y);
    }
};

struct Add
{
    template<typename scalar_t>
    __device__ __forceinline__ double operator()(double x, scalar_t y) const {
        return x + y;
    }
};


struct SumExp
{
    __device__ __forceinline__ SumExp(double v) : max_k(v) {}

    template<typename scalar_t>
    __device__ __forceinline__ double operator()(double sum, scalar_t v) const {
        return sum + static_cast<double>(exp((double)v - max_k));
    }
    
    const double max_k;
};


template <typename Reduction>
__device__ __forceinline__ double
blockReduce(double* smem, double val,
            const Reduction& r,
            double defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  double warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  double blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}


template <typename Reduction, int ILP, typename scalar_t>
__device__ __forceinline__ double
ilpReduce(scalar_t* data,
          int size,
          const Reduction& r,
          double defaultVal)
{
  double threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      threadVal = r(threadVal, tmp[j]);
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}


template <int ILP, typename scalar_t>
__global__ void cunn_SoftMaxForward(scalar_t *output, scalar_t *input, int64_t* ps)
{
    SharedMem smem;
    double *buffer = smem.getPointer();
    // forward pointers to batch[blockIdx.x]
    // each block handles a sample in the mini-batch
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    int64_t n_ele = ps[blockIdx.x] - ofs;
    input += ofs;
    output += ofs;
    
    // find the max
    double threadMax = ilpReduce<Max, ILP, scalar_t>(input, n_ele, Max(), -DBL_MAX);
    double max_k = blockReduce<Max>(buffer, threadMax, Max(), -DBL_MAX);
    
    // reduce all values
    double threadExp = ilpReduce<SumExp, ILP, scalar_t>(input, n_ele, SumExp(max_k), static_cast<double>(0));
    
    double sumAll = blockReduce<Add>(buffer, threadExp, Add(), static_cast<double>(0));
    double logsum = max_k + log(sumAll);
    
    int offset = threadIdx.x;
    int last = n_ele % (ILP * blockDim.x);
    for (; offset < n_ele - last; offset += blockDim.x * ILP) {
        scalar_t tmp[ILP];
        
        #pragma unroll
        for (int j = 0; j < ILP; ++j)
            tmp[j] = input[offset + j * blockDim.x];
        
        #pragma unroll
        for (int j = 0; j < ILP; ++j)
            output[offset + j * blockDim.x] = (double)tmp[j] - logsum;
    }
    
    for (; offset < n_ele; offset += blockDim.x)
        output[offset] = (double)input[offset] - logsum;
}


template<typename scalar_t>
void HostLogSoftmaxForward(scalar_t* input, scalar_t *output, int64_t* ps, int64_t bsize)
{
    dim3 grid(bsize);
    dim3 block(1024);
    
    cunn_SoftMaxForward<2>
    <<<grid, block, block.x * sizeof(double)>>>(
        output, input, ps
    );
}

template void HostLogSoftmaxForward<float>(float* input, float* output, int64_t* ps, int64_t bsize);
template void HostLogSoftmaxForward<double>(double* input, double* output, int64_t* ps, int64_t bsize);

template <int ILP, typename scalar_t>
__global__ void cunn_SoftMaxBackward(scalar_t *gradInput, scalar_t *output, scalar_t *gradOutput, int64_t* ps)
{
    SharedMem smem;
    double *buffer = smem.getPointer();
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    int64_t n_ele = ps[blockIdx.x] - ofs;
    
    gradInput += ofs;
    output += ofs;
    gradOutput += ofs;
    
    double threadSum = ilpReduce<Add, 4>(gradOutput, n_ele, Add(), double(0));
    double sum_k = blockReduce<Add>(buffer, threadSum, Add(), double(0));
    
    int offset = threadIdx.x;
    int last = n_ele % (ILP * blockDim.x);
    for (; offset < n_ele - last; offset += blockDim.x * ILP) {
        scalar_t tmpGradOutput[ILP];
        scalar_t tmpOutput[ILP];

        #pragma unroll
        for (int j = 0; j < ILP; ++j) {
            tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
            tmpOutput[j] = output[offset + j * blockDim.x];
        }
        
        #pragma unroll
        for (int j = 0; j < ILP; ++j)
            gradInput[offset + j * blockDim.x] = tmpGradOutput[j] - exp((double)tmpOutput[j]) * sum_k;
    }

    for (; offset < n_ele; offset += blockDim.x)
        gradInput[offset] = gradOutput[offset] - exp((double)output[offset]) * sum_k;
}


template<typename scalar_t>
void HostLogSoftmaxBackward(scalar_t *gradOutput, scalar_t *gradInput, scalar_t *output, int64_t* ps, int64_t bsize)
{
    dim3 grid(bsize);
    dim3 block(1024);
  
    cunn_SoftMaxBackward<2>
    <<<grid, block, block.x * sizeof(double)>>>(
        gradInput, output, gradOutput, ps
    );
}

template void HostLogSoftmaxBackward<float>(float *gradOutput, float *gradInput, float *output, int64_t* ps, int64_t bsize);
template void HostLogSoftmaxBackward<double>(double *gradOutput, double *gradInput, double *output, int64_t* ps, int64_t bsize);
