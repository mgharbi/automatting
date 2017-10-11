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



__global__ void spadd_backward_matrix_kernel(
      const int* p_csr_rowA, const int* p_csr_colA, float* p_gradA, const int nnzA,
      const int* p_csr_rowB, const int* p_csr_colB, float* p_gradB, const int nnzB,
      const int* p_coo_rowC, const int* p_csr_colC, const float* p_gradC, const int nnzC,
      const float alpha, const float beta, const int rows, const int cols) {
  const int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < nnzC) {
    int row = p_coo_rowC[idx];
    int col = p_csr_colC[idx];

    int ptrA = p_csr_rowA[row];
    int endA = p_csr_rowA[row+1];
    while(ptrA < endA && p_csr_colA[ptrA] < col) {
      ++ptrA;
    }

    if (ptrA < endA && p_csr_colA[ptrA] == col) {
      // update gradient
      p_gradA[ptrA] = p_gradC[idx];
    }

    int ptrB = p_csr_rowB[row];
    int endB = p_csr_rowB[row+1];
    while(ptrB < endB && p_csr_colB[ptrB] < col) {
      ++ptrB;
    }

    if (ptrB < endB && p_csr_colB[ptrB] == col) {
      // update gradient
      p_gradB[ptrB] = p_gradC[idx];
    }

  }
}


void spadd_backward_matrix_cuda(
      const int* p_csr_rowA, const int* p_csr_colA, float* p_gradA, const int nnzA,
      const int* p_csr_rowB, const int* p_csr_colB, float* p_gradB, const int nnzB,
      const int* p_coo_rowC, const int* p_csr_colC, const float* p_gradC, const int nnzC,
      const float alpha, const float beta, const int rows, const int cols) {

  const int64_t block_sz = 512;
  const int64_t nblocks = (nnzC + block_sz - 1) / block_sz;

  spadd_backward_matrix_kernel<<<nblocks, block_sz, 0, THCState_getCurrentStream(state)>>>(
       p_csr_rowA, p_csr_colA, p_gradA, nnzA,
       p_csr_rowB, p_csr_colB, p_gradB, nnzB,
       p_coo_rowC, p_csr_colC, p_gradC, nnzC,
       alpha, beta, rows, cols);
  THCudaCheck(cudaPeekAtLastError());
}
