#include <stdio.h>

__global__
void matrixAdd(int rows, int cols, float *x, float *y, float *z)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  int flatId = rows*row + col;

  if(row < rows && col < cols)
  {
    z[flatId] = x[flatId] + y[flatId];
  }
}

int main(void)
{
  // define timing variables
  struct timespec start, end; 
  double cpu_time_used, gpu_time_used;
  long seconds, nanoseconds;

  // define matrix size
  int rows = 1024;
  int cols = 1024;
  int N = rows*cols;
  printf("Data Size: %d\n", N);
  printf("Rows: %d\n", rows);
  printf("Columns: %d\n", cols);

  // create pointers
  float *x, *y, *z_gpu, *z_cpu, *d_x, *d_y, *d_z;

  // allocate cpu memory to cpu pointers
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  z_gpu = (float*)malloc(N*sizeof(float));
  z_cpu = (float*)malloc(N*sizeof(float));

  // allocate gpu memory to gpu pointers
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_z, N*sizeof(float));

  // assign values to cpu arrays
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // ***start profile operation on CPU***
  clock_gettime(CLOCK_REALTIME, &start);
  for(int i = 0; i < N; i++)
  {
    z_cpu[i] = x[i] + y[i];
  }
  clock_gettime(CLOCK_REALTIME, &end);
  seconds = end.tv_sec - start.tv_sec;
  nanoseconds = end.tv_nsec - start.tv_nsec;
  cpu_time_used = seconds + nanoseconds*1e-9;
  printf("CPU Time: %f\n", cpu_time_used);
  // ***end profile operation on CPU***

  // copy memory from CPU to GPU
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // calculate block/grid dimensions
  dim3 blockSize(32, 32, 1);
  dim3 gridSize(cols/blockSize.x,
                rows/blockSize.y,
                1);

  // ***start profile operation on GPU***
  clock_gettime(CLOCK_REALTIME, &start);

  // call kernel
  matrixAdd<<<gridSize, blockSize>>>(rows, cols, d_x, d_y, d_z);

  clock_gettime(CLOCK_REALTIME, &end);
  seconds = end.tv_sec - start.tv_sec;
  nanoseconds = end.tv_nsec - start.tv_nsec;
  gpu_time_used = seconds + nanoseconds*1e-9;
  printf("GPU Time: %f\n", gpu_time_used);
  if(gpu_time_used < cpu_time_used)
  {
    printf("\033[1;32mGPU is %f times faster than CPU!\033[0m\n", cpu_time_used/gpu_time_used);
  }
  else
  {
    printf("\033[1;31mGPU is %f times slower than CPU!\033[0m\n", cpu_time_used/gpu_time_used);
  }
  // ***end profile operation on GPU***

  // copy result back
  cudaMemcpy(z_gpu, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

  // run through result on cpu and show that results is correct
  float maxError = 0.0f;
  float expectedVal = 3.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(z_gpu[i]-expectedVal));
  printf("Max error: %f\n", maxError);

  // free GPU memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  // free CPU memory
  free(x);
  free(y);
  free(z_gpu);
  free(z_cpu);
}