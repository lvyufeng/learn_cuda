#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void){
    printf("hello word! form thread {%d, %d} from device\n", threadIdx.x, blockIdx.x);
}

int main(void){
    printf("hello word from host\n");
    print_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}