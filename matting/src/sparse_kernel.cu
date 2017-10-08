#include <THC/THC.h>
#include "sparse_kernel.h"

extern THCState *state;

__global__ void spmv_backward_matrix_kernel(
    const int* p_cooRow, const int* p_csrCol, const float* p_vector,
    const float* p_grad_output, float* p_grad_matrix,
    const int rows, const int cols, const int nnz) {
  const int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nnz) {
    int row = p_cooRow[idx];
    int col = p_csrCol[idx];
    p_grad_matrix[idx] = p_grad_output[row]*p_vector[col];
  }
}


void spmv_backward_matrix_cuda(
    const int* p_cooRow, const int* p_csrCol, const float* p_vector,
    const float* p_grad_output, float* p_grad_matrix,
    const int rows, const int cols, const int nnz) {

  const int64_t block_sz = 512;
  const int64_t nblocks = (nnz + block_sz - 1) / block_sz;
  spmv_backward_matrix_kernel<<<nblocks, block_sz, 0, THCState_getCurrentStream(state)>>>(
      p_cooRow, p_csrCol, p_vector, p_grad_output, p_grad_matrix, 
      rows, cols, nnz);
  THCudaCheck(cudaPeekAtLastError());
}
