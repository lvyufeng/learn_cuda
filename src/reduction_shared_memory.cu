#include<stdio.h>
#define THREADS 1024
/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// cuda thread synchronization
__global__ void
reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // thread synchronous reduction
        if ( (idx_x % (stride * 2)) == 0 )
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_out[blockIdx.x] = s_data[0];
}

void reduction(float *d_out, float *d_in, int size, cudaStream_t stream)
{   
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
    while(size > 1)
    {
        int n_blocks = (size + THREADS - 1) / THREADS;
        reduction_kernel<<< n_blocks, THREADS, THREADS * sizeof(float), stream >>>(d_out, d_out, size);
        size = n_blocks;
    } 
}

extern "C" int ReductionNative(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                               void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 2) return 1;
  void *input = params[0];
  void *output = params[1];
  size_t size = 1;

  for (int i = 0; i < ndims[0]; i++) {
    size *= shapes[0][i];
  }

  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  reduction(static_cast<float *>(output), static_cast<float *>(input), size, custream);
  return 0;
}