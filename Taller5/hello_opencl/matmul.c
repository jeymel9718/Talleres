#include <stdio.h>
#include <time.h>
#include <stdlib.h>
// Include OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
// What's up with this? :O
#define MAX_SOURCE_SIZE (0x100000)

void mat_mul(int* A,int* B,int *C){
  for (int m=0; m<4; m++) {
    for (int n=0; n<4; n++) {
        int acc = 0;
        for (int k=0; k<4; k++) {
            acc += A[k*4 + m] * B[n*4 + k];
        }
        C[n*4 + m] = acc;
    }
  }
}

int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 16;
    clock_t start_t,end_t;
    double time_p,time_s;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    // Init the vectors
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    // Read the kernel
    fp = fopen("matmul.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    //1. Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
            &device_id, &ret_num_devices);

    //2. Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

    //3. Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    //4. Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "mat_mul_par", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    start_t = clock();
    //5. Execute the OpenCL kernel on the list
    size_t global_item_size[2] = {4,4}; // Process the entire lists
    size_t local_item_size[2] = {4,4}; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
            &global_item_size, &local_item_size, 0, NULL, NULL);

    end_t = clock();
    time_p = (double)(end_t-start_t)/ CLOCKS_PER_SEC;
    // Read the memory buffer C on the device to the local variable C
    int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
    start_t = clock();
    mat_mul(A,B,C);
    end_t = clock();
    time_s = (double)(end_t-start_t)/ CLOCKS_PER_SEC;
    //6. Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    // Be nice
    printf("Multiplicacion paralela: %f\nMultiplicacion serial: %f\n",time_p,time_s);
    return 0;
}
