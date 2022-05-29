#include<stdio.h>
#define THREADS 1024
__global__ void
global_reduction_kernel(float *data_out, float *data_in, int stride, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_x + stride < size) {
        data_out[idx_x] += data_in[idx_x + stride];
        printf("%d %d %f\n", idx_x, stride, data_out[idx_x]);
    }
}

void global_reduction(float *d_out, float *d_in, int n_threads, int size)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    for (int stride = 1; stride < size; stride *= 2) {
        global_reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, stride, size);
    }
}

extern "C" int ReductionNative(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                               void *extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  if (nparam != 2) return 1;
  void *input = params[0];
  void *output = params[1];
  size_t size = 1;

  for (int i = 0; i < ndims[1]; i++) {
    size *= shapes[1][i];
  }
  int n = size / THREADS;
  printf("size: %d \n", size);
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }
  global_reduction(static_cast<float *>(output),
                  static_cast<float *>(input),
                  THREADS, size);

  return 0;
}