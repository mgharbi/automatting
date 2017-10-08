#include <THC/THC.h>
#include <stdio.h>

#include "sparse_kernel.h"

extern THCState *state;

void sortCOOMatrix(
    const int rows, const int cols, const int nnz,
    int* p_cooRow, int* p_cooCol, float* p_cooVal) {
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseStatus_t status; 
  size_t pBufferSizeInBytes;
  status = cusparseXcoosort_bufferSizeExt(
      handle, rows, cols,
      nnz, p_cooRow, p_cooCol, &pBufferSizeInBytes);
  THCusparseCheck(status);

  int* permutation;
  THCudaCheck(THCudaMalloc(state, (void**) &permutation, nnz*sizeof(int)));
  int* pBuffer;
  THCudaCheck(THCudaMalloc(state, (void**) &pBuffer, pBufferSizeInBytes*sizeof(char)));
  float* sortedVals;
  THCudaCheck(THCudaMalloc(state, (void**) &sortedVals, nnz*sizeof(float)));

  THCusparseCheck(cusparseCreateIdentityPermutation(handle, nnz, permutation));
  THCusparseCheck(cusparseXcoosortByRow(handle, rows, cols, nnz, p_cooRow, p_cooCol, permutation, pBuffer));
  THCusparseCheck(cusparseSgthr(handle, nnz, p_cooVal, sortedVals, permutation, CUSPARSE_INDEX_BASE_ZERO));
  THCudaCheck(cudaMemcpy(p_cooVal, sortedVals, nnz*sizeof(float), cudaMemcpyDeviceToDevice));

  THCudaCheck(THCudaFree(state, sortedVals));
  THCudaCheck(THCudaFree(state, pBuffer));
  THCudaCheck(THCudaFree(state, permutation));
}


int coo2csr(THCudaIntTensor *row_idx, 
            THCudaIntTensor *col_idx,
            THCudaTensor *val,
            THCudaIntTensor *csr_row_idx,
            const int rows, const int cols) {

  THArgCheck(THCudaIntTensor_nDimension(state, row_idx) == 1, 0, "row_idx should be 1D");
  THArgCheck(THCudaIntTensor_nDimension(state, col_idx) == 1, 1, "col_idx should be 1D");

  if( THCudaTensor_nDimension(state, val) != 1) {
    THError("val should be 1D");
    return 1;
  }
  int nnz = THCudaTensor_size(state, val, 0);
  if( THCudaIntTensor_size(state, col_idx, 0) != nnz) {
    THError("row_idx and col_idx should have matching nnz.");
    return 1;
  }
  if( THCudaTensor_size(state, val, 0) != nnz) {
    THError("idx and val should have matching nnz.");
    return 1;
  }
  if(nnz > rows*cols) {
    THError("nnz is higher than rows*cols");
    return 1;
  }

  int *p_cooRow = THCudaIntTensor_data(state, row_idx);
  int *p_cooCol = THCudaIntTensor_data(state, col_idx);
  float *p_cooVal = THCudaTensor_data(state, val);

  sortCOOMatrix(rows, cols, nnz, p_cooRow, p_cooCol, p_cooVal);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  THCudaIntTensor_resize1d(state, csr_row_idx, rows+1);
  int *p_csr_row_idx = THCudaIntTensor_data(state, csr_row_idx);

  THCusparseCheck(cusparseXcoo2csr(
      handle, p_cooRow, nnz, rows, p_csr_row_idx, CUSPARSE_INDEX_BASE_ZERO));

  return 0;
}


int spadd_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *A_val,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *B_val,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_val,
    const float alpha, const float beta, const int rows, const int cols) {

  int nnzA = THCudaTensor_size(state, A_val, 0);
  int nnzB = THCudaTensor_size(state, B_val, 0);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  // Setup
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int *p_rowA = THCudaIntTensor_data(state, A_csr_row);
  int *p_colA = THCudaIntTensor_data(state, A_csr_col);
  float *p_valA = THCudaTensor_data(state, A_val);

  int *p_rowB = THCudaIntTensor_data(state, B_csr_row);
  int *p_colB = THCudaIntTensor_data(state, B_csr_col);
  float *p_valB = THCudaTensor_data(state, B_val);

  THCudaIntTensor_resize1d(state, C_csr_row, rows+1);
  int *p_rowC = THCudaIntTensor_data(state, C_csr_row);

  int nnzC;
  int* nnzTotalDevHostPtr = &nnzC;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);  // nnzTotalDevHostPtr points to host memory
  THCusparseCheck(cusparseXcsrgeamNnz(
      handle, rows, cols,
      descr, nnzA, p_rowA, p_colA,
      descr, nnzB, p_rowB, p_colB,
      descr, p_rowC, nnzTotalDevHostPtr));

  if(NULL != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    int baseC;
    THCudaCheck(cudaMemcpy(&nnzC, p_rowC+rows, sizeof(int), cudaMemcpyDeviceToHost));
    THCudaCheck(cudaMemcpy(&baseC, p_rowC, sizeof(int), cudaMemcpyDeviceToHost));
    nnzC -= baseC;
  }

  THCudaIntTensor_resize1d(state, C_csr_col, nnzC);
  THCudaTensor_resize1d(state, C_val, nnzC);
  int *p_colC = THCudaIntTensor_data(state, C_csr_col);
  float *p_valC = THCudaTensor_data(state, C_val);

  THCusparseCheck(cusparseScsrgeam(
      handle, rows, cols,
      &alpha, descr, nnzA, p_valA, p_rowA, p_colA,
      &beta, descr, nnzB, p_valB, p_rowB, p_colB,
      descr, p_valC, p_rowC, p_colC));

  return 0;
}


int spmv_forward(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col, THCudaTensor *val,
    THCudaTensor *vector,
    THCudaTensor *output,
    const int rows, const int cols, const int transpose) {

  int nnz = THCudaTensor_size(state, val, 0);

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, csr_row, csr_col, val, vector, output));

  csr_row = THCudaIntTensor_newContiguous(state, csr_row);
  csr_col = THCudaIntTensor_newContiguous(state, csr_col);
  val = THCudaTensor_newContiguous(state, val);
  vector = THCudaTensor_newContiguous(state, vector);

  THArgCheck(rows+1 == THCudaIntTensor_size(state, csr_row, 0), 0,
      "csr rows should have rows+1 entries");
  THArgCheck(nnz == THCudaIntTensor_size(state, csr_col, 0), 1,
      "csr cols should have nnz entries");

  int vector_size = THCudaTensor_size(state, vector, 0);
  if(transpose == 1) {
    THArgCheck(rows == vector_size,
        3, "rows should match vector size in transpose mode got %d expected %d", rows, vector_size);
    THCudaTensor_resize1d(state, output, cols);
  } else {
    THArgCheck(cols == vector_size,
        3, "cols should match vector size in non-transpose mode");
    THCudaTensor_resize1d(state, output, rows);
  }
  THCudaTensor_zero(state, output);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  // Setup
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int *p_row = THCudaIntTensor_data(state, csr_row);
  int *p_col = THCudaIntTensor_data(state, csr_col);
  float *p_val = THCudaTensor_data(state, val);
  float *p_vector = THCudaTensor_data(state, vector);

  float *p_output = THCudaTensor_data(state, output);

  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  if(transpose == 1) {
    trans = CUSPARSE_OPERATION_TRANSPOSE;
  }

  /* TODO(mgharbi): more accurate version when transposing: */
  /* convert to CSC and run with NON_TRANSPOSE. */
  float multiplier = 1.0f;
  THCusparseCheck(cusparseScsrmv(handle, trans,
        rows, cols, nnz, &multiplier, descr, p_val, p_row, p_col,
        p_vector, &multiplier, p_output));

  THCudaIntTensor_free(state, csr_row);
  THCudaIntTensor_free(state, csr_col);
  THCudaTensor_free(state, val);
  THCudaTensor_free(state, vector);
  return 0;
}


int spmm_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *A_val,
    const int rowsA, const int colsA, int transposeA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *B_val,
    const int rowsB, const int colsB, int transposeB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_val) {

  THAssertMsg(colsA == rowsB, "spmm: A and B should have compatible inner dimensions.");

  int nnzA = THCudaTensor_size(state, A_val, 0);
  int nnzB = THCudaTensor_size(state, B_val, 0);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  // Setup
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int *p_rowA = THCudaIntTensor_data(state, A_csr_row);
  int *p_colA = THCudaIntTensor_data(state, A_csr_col);
  float *p_valA = THCudaTensor_data(state, A_val);

  int *p_rowB = THCudaIntTensor_data(state, B_csr_row);
  int *p_colB = THCudaIntTensor_data(state, B_csr_col);
  float *p_valB = THCudaTensor_data(state, B_val);

  THCudaIntTensor_resize1d(state, C_csr_row, rowsA+1);
  int *p_rowC = THCudaIntTensor_data(state, C_csr_row);

  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  /* if(transposeA == 1) { */
  /*   transA = CUSPARSE_OPERATION_TRANSPOSE; */
  /* } */
  /* if(transposeB == 1) { */
  /*   transB = CUSPARSE_OPERATION_TRANSPOSE; */
  /* } */

  int nnzC;
  int* nnzTotalDevHostPtr = &nnzC;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);  // nnzTotalDevHostPtr points to host memory
  THCusparseCheck(cusparseXcsrgemmNnz(
      handle, transA, transB, rowsA, rowsB, colsA,
      descr, nnzA, p_rowA, p_colA,
      descr, nnzB, p_rowB, p_colB,
      descr, p_rowC, nnzTotalDevHostPtr));

  if(NULL != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    int baseC;
    THCudaCheck(cudaMemcpy(&nnzC, p_rowC+rowsA, sizeof(int), cudaMemcpyDeviceToHost));
    THCudaCheck(cudaMemcpy(&baseC, p_rowC, sizeof(int), cudaMemcpyDeviceToHost));
    nnzC -= baseC;
  }

  THCudaIntTensor_resize1d(state, C_csr_col, nnzC);
  THCudaTensor_resize1d(state, C_val, nnzC);
  int *p_colC = THCudaIntTensor_data(state, C_csr_col);
  float *p_valC = THCudaTensor_data(state, C_val);

  THCusparseCheck(cusparseScsrgemm(
      handle, transA, transB, rowsA, rowsB, colsA,
      descr, nnzA, p_valA, p_rowA, p_colA,
      descr, nnzB, p_valB, p_rowB, p_colB,
      descr, p_valC, p_rowC, p_colC));

  return 0;
}


int spmv_backward_matrix(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col,
    THCudaTensor *vector,
    THCudaTensor *grad_output, THCudaTensor *grad_matrix,
    const int rows, const int cols) {

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  int nnz = THCudaIntTensor_size(state, csr_col, 0);
  int *p_csrRow = THCudaIntTensor_data(state, csr_row);
  int *p_csrCol = THCudaIntTensor_data(state, csr_col);
  /* float *p_cooVal = THCudaTensor_data(state, val); */

  int* p_cooRow;
  THCudaCheck(THCudaMalloc(state, (void**) &p_cooRow, nnz*sizeof(int)));
  THCusparseCheck(cusparseXcsr2coo(
      handle, p_csrRow, nnz, rows, p_cooRow, CUSPARSE_INDEX_BASE_ZERO));

  THCudaTensor_resize1d(state, grad_matrix, nnz);
  THCudaTensor_zero(state, grad_matrix);

  float* p_vector = THCudaTensor_data(state, vector);
  float* p_grad_output = THCudaTensor_data(state, grad_output);
  float* p_grad_matrix = THCudaTensor_data(state, grad_matrix);
  spmv_backward_matrix_cuda(p_cooRow, p_csrCol, p_vector, p_grad_output, p_grad_matrix, rows, cols);

  THCudaCheck(THCudaFree(state, p_cooRow));

  return 0;
}
