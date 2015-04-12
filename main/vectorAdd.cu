#include<iostream>

#define SIZE 2048
#define THREADS_PER_BLOCK 512

using std::cout;
using std::endl;

// __global__ indicates that this function runs on the GPU but is
// called from the cpu code.
//
// nvcc splits things up into shit that runs on the device and
// things that run on the host. device=gpu, host=cpu.
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
  // a, b, and c point to device memory now. How do we allocate
  // device memory?
  //
  // Host pointers can be passed to device code but cannot
  // be dereferenced by device code. The same is true for host
  // pointers-- they can't be dereferenced by device code.
  //
  // So basically what you do is you let the GPU do shit in its
  // own memory, then copy everything over at the end.
  //
  // Basic CUDA API for dealing with device memory
  // — cudaMalloc(), cudaFree(), cudaMemcpy()
  // — Similar to malloc(), free(), memcpy()
  //
  // Ok, so we're using threads and blocks. A block is a GPU processor
  // basically and each block has some number of threads. In order
  // to split up the work, we use the threaad index (threadIdx), the
  // block index (blockkIdx), and the number of threads per block
  // (blockDim). If you're still confused, read this whole thing:
  //
  // http://www.nvidia.com/content/gtc-2010/pdfs/2131_gtc2010.pdf
  //
  // Basically the reason threads are exposed as opposed to being hardcore
  // abstracted away is that threads can communicate with each other
  // and shit so they offer some advantages over just splitting everything
  // up by block.
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index > n) {
    return;
  }
  c[index] = b[index] + a[index];
}

int main() {
  int *a, *b, *c;

  // Shit that runs on our device needs to deal with stuff
  // allocated in its own memory space. The pointers below
  // are going to be copies of abc that the device has
  // access to. People usually prefix device-side pointers
  // with d_ as a convention.
  int *d_a, *d_b, *d_c;

  a = (int*)malloc(SIZE * sizeof(int));
  b = (int*)malloc(SIZE * sizeof(int));
  c = (int*)malloc(SIZE * sizeof(int));

  // Just like we use malloc to allocate host-side memory,
  // we use cudaMalloc to allocate device-side memory.
  // cudaMalloc needs a pointer to our array and I think
  // it's because it wants to write 0 to it or something
  // when an error occurs.
  cudaMalloc((void**) &d_a, SIZE * sizeof(int));
  cudaMalloc((void**) &d_b, SIZE * sizeof(int));
  cudaMalloc((void**) &d_c, SIZE * sizeof(int));

  for (int i = 0; i < SIZE; i++) {
    a[i] = i;
    b[i] = i;
    c[i] = 0;
  }

  // We need to copy our host-side arrays into our device-side
  // memory.
  cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, SIZE*sizeof(int), cudaMemcpyHostToDevice);

  // The brackets allows you to call a function meant to run on a
  // device. The left number is blocks, the right number is threads.
  // Don't ask me what the difference is...
  //
  // This sortof explains but it's still somewhat confusing...
  // http://bit.ly/1A9Pww8
  int NUM_BLOCKS = SIZE / THREADS_PER_BLOCK + 1;
  vectorAdd<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, SIZE);

  // Now we need to copy the device-side results to the host-side
  // arrays.
  cudaMemcpy(a, d_a, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, d_b, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  double c_sum = 0;
  for (int i = 0; i < SIZE; i++) {
    c_sum += c[i];
  }
  cout << "c arr result was: " << c_sum << endl;
  cout << "EXPECTED: " << (double)SIZE*(SIZE-1) << endl;;
  if (c_sum == (double)SIZE*(SIZE-1)) {
    cout << "TEST PASSED" << endl;
  } else {
    cout << "TEST FAILED" << endl;
  }

  free(a);
  free(b);
  free(c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
