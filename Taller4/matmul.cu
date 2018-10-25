#include <stdio.h>
#include <cuda.h>
#include <math.h>

int *a, *b;  // host data
int *c, *c2;  // results

//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0);
 }
}

//GPU kernel
__global__
void matMul(int *A,int *B,int *C,int N){
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    float tmpSum = 0;
    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
     C[ROW*N + COL] = tmpSum;
    }
}

//CPU function
void matMul_h(int *A1,int *B1, int *C1, int N){
    float tmpSum=0;
    for(int i = 0; i < N; i++){
      for(int j = 0; j < N;++j){
       for (int k = 0; k < N;++k){
         tmpSum+=A1[i*N + k] + B1[k*N + j];
       }
       C1[i*N + j] = tmpSum;
       tmpSum = 0;
      }
    }
}

int main(int argc,char **argv)
{
    printf("Begin \n");
    //N*N matrix
    int n=4*4;
    //Number of blocks
    int nBytes = n*sizeof(int);
    //Block size and number
    int block_size, block_no;

    //memory allocation
    a = (int *) malloc(nBytes);
    b = (int *) malloc(nBytes);
    c = (int *) malloc(nBytes);
    c2 = (int *) malloc(nBytes);

    int *a_d,*b_d,*c_d;
    block_size = 4; //threads per block
    block_no = 1;

    //Work definition
    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(block_no, block_no, 1);

    // Data filling
    for(int i=0;i<n;i++)
    a[i]=i,b[i]=i;


    printf("Allocating device memory on host..\n");
   //GPU memory allocation
    cudaMalloc((void **) &a_d, n*sizeof(int));
    cudaMalloc((void **) &b_d, n*sizeof(int));
    cudaMalloc((void **) &c_d, n*sizeof(int));

    printf("Copying to device..\n");
    cudaMemcpy(a_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_d=clock();
    printf("Doing GPU Vector add\n");
    matMul<<<block_no,block_size>>>(a_d, b_d, c_d, n);
    cudaCheckError();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();


    printf("Doing CPU Vector add\n");
    clock_t start_h = clock();
    matMul_h(a, b, c2, n);
    clock_t end_h = clock();

    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    //Copying data back to host, this is a blocking call and will not start unt$
    cudaMemcpy(c, c_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("n = %d \t GPU time = %fs \t CPU time = %fs\n", n, time_d, time_h);

    //Free GPU memory
     cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}




