#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" int SGEMM(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                     void *extra) {
    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    cublasHandle_t handle;
    // create cuBLAS handle
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    cublasSetStream(handle, custream);

    if (nparam != 3) return 1;
    void *A = params[0];
    void *B = params[1];
    void *C = params[2];
    size_t size = 1;

    float alpha = 1.f;
    float beta = 0.f;

    int M = shapes[0][1];
    int N = shapes[1][0];
    int K = shapes[1][1];
    for (int i = 0; i < ndims[2]; i++) {
        size *= shapes[2][i];
    }

    for (int i = 0; i < nparam; i++) {
        if (strcmp(dtypes[i], "float32") != 0) {
            return 2;
        }
    }

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                M, N, K, &alpha, 
                static_cast<float *>(A), K,
                static_cast<float *>(B), N,
                &beta,
                static_cast<float *>(C), M
                );
    cudaDeviceSynchronize();
    cublasDestroy(handle);

    return 0;
}