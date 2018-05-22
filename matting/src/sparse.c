#include <THC/THC.h>
#include <stdio.h>

#include "sparse_kernel.h"

extern THCState *state;


void sortCOOMatrix(
    const long rows, const long cols, const long nnz,
    int* p_cooRow, int* p_cooCol, double* p_cooVal, double* p_sorted_val, int* permutation) {
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseStatus_t status;

  // Find working buffer size
  size_t pBufferSizeInBytes;
  status = cusparseXcoosort_bufferSizeExt(
      handle, rows, cols,
      nnz, p_cooRow, p_cooCol, &pBufferSizeInBytes);
  THCusparseCheck(status);

  // Allocate
  int* pBuffer;
  THCudaCheck(
      THCudaMalloc(state, (void**) &pBuffer, pBufferSizeInBytes*sizeof(char)));
  /* double* sortedVals; */
  /* THCudaCheck(THCudaMalloc(state, (void**) &sortedVals, nnz*sizeof(double))); */

  THCusparseCheck(cusparseCreateIdentityPermutation(
        handle, nnz, permutation));
  THCusparseCheck(cusparseXcoosortByRow(
        handle, rows, cols, nnz, p_cooRow, p_cooCol, permutation, pBuffer));
  THCusparseCheck(cusparseDgthr(
        handle, nnz, p_cooVal, p_sorted_val,
        permutation, CUSPARSE_INDEX_BASE_ZERO));
  /* THCudaCheck(cudaMemcpy( */
  /*       p_cooVal, sortedVals, nnz*sizeof(double), cudaMemcpyDeviceToDevice)); */

  /* THCudaCheck(THCudaFree(state, sortedVals)); */
  THCudaCheck(THCudaFree(state, pBuffer));
}


int coo2csr(THCudaIntTensor *row_idx,
            THCudaIntTensor *col_idx,
            THCudaDoubleTensor *val,
            THCudaIntTensor *csr_row_idx,
            THCudaIntTensor *csr_col_idx,
            THCudaDoubleTensor *csr_val,
            THCudaIntTensor *permutation,
            const long rows, const long cols) {

  THArgCheck(THCudaIntTensor_nDimension(state, row_idx) == 1,
                                        0, "row_idx should be 1D");
  THArgCheck(THCudaIntTensor_nDimension(state, col_idx) == 1,
                                        1, "col_idx should be 1D");

  // Grab reference
  row_idx = THCudaIntTensor_newContiguous(state, row_idx);
  col_idx = THCudaIntTensor_newContiguous(state, col_idx);
  val = THCudaDoubleTensor_newContiguous(state, val);

  if( THCudaDoubleTensor_nDimension(state, val) != 1) {
    THError("val should be 1D");
    return 1;
  }
  long nnz = THCudaDoubleTensor_size(state, val, 0);
  if( THCudaIntTensor_size(state, col_idx, 0) != nnz) {
    THError("row_idx and col_idx should have matching nnz.");
    return 1;
  }
  if( THCudaDoubleTensor_size(state, val, 0) != nnz) {
    THError("idx and val should have matching nnz.");
    return 1;
  }
  if(nnz > rows*cols) {
    THError("nnz is higher than rows*cols");
    return 1;
  }

  THCudaIntTensor_resize1d(state, csr_col_idx, nnz);
  THCudaDoubleTensor_resize1d(state, csr_val, nnz);
  THCudaIntTensor_resize1d(state, permutation, nnz);

  THCudaIntTensor_zero(state, permutation);

  double *p_cooVal = THCudaDoubleTensor_data(state, val);

  int *p_csrCol = THCudaIntTensor_data(state, csr_col_idx);
  double *p_csrVal = THCudaDoubleTensor_data(state, csr_val);
  int *p_permutation = THCudaIntTensor_data(state, permutation);

  // The sort is in place so we allocate new output buffer with copies of the indices
  int *p_cooRow   = THCudaIntTensor_data(state, row_idx);
  int *p_cooRow_copy;
  THCudaCheck(THCudaMalloc(state, (void**) &p_cooRow_copy, nnz*sizeof(double)));
  THCudaCheck(cudaMemcpy(
        p_cooRow_copy, p_cooRow, nnz*sizeof(int), cudaMemcpyDeviceToDevice));
  int *p_cooCol   = THCudaIntTensor_data(state, col_idx);
  THCudaCheck(cudaMemcpy(
        p_csrCol, p_cooCol, nnz*sizeof(int), cudaMemcpyDeviceToDevice));

  sortCOOMatrix(rows, cols, nnz, p_cooRow_copy, p_csrCol, p_cooVal, p_csrVal, p_permutation);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  THCudaIntTensor_resize1d(state, csr_row_idx, rows+1);
  int *p_csr_row_idx = THCudaIntTensor_data(state, csr_row_idx);

  THCusparseCheck(cusparseXcoo2csr(
      handle, p_cooRow_copy, nnz, rows, p_csr_row_idx, CUSPARSE_INDEX_BASE_ZERO));

  THCudaCheck(THCudaFree(state, p_cooRow_copy));

  THCudaIntTensor_free(state, row_idx);
  THCudaIntTensor_free(state, col_idx);
  THCudaDoubleTensor_free(state, val);

  return 0;
}

int csr2csc(THCudaIntTensor *row_idx,
            THCudaIntTensor *col_idx,
            THCudaDoubleTensor *val,
            THCudaIntTensor *csc_row_idx,
            THCudaIntTensor *csc_col_idx,
            THCudaDoubleTensor *csc_val,
            const long rows, const long cols) {

  int nnz = THCudaIntTensor_size(state, col_idx, 0);

  row_idx = THCudaIntTensor_newContiguous(state, row_idx);
  col_idx = THCudaIntTensor_newContiguous(state, col_idx);
  val = THCudaDoubleTensor_newContiguous(state, val);
  int* p_row = THCudaIntTensor_data(state, row_idx);
  int* p_col = THCudaIntTensor_data(state, col_idx);
  double* p_val = THCudaDoubleTensor_data(state, val);

  THCudaIntTensor_resize1d(state, csc_row_idx, nnz);
  THCudaIntTensor_resize1d(state, csc_col_idx, cols+1);
  THCudaDoubleTensor_resize1d(state, csc_val, nnz);
  int* p_csc_row = THCudaIntTensor_data(state, csc_row_idx);
  int* p_csc_col = THCudaIntTensor_data(state, csc_col_idx);
  double* p_csc_val = THCudaDoubleTensor_data(state, csc_val);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  THCusparseCheck(cusparseDcsr2csc(
      handle, rows, cols, nnz,
      p_val, p_row, p_col,
      p_csc_val, p_csc_row, p_csc_col,
      CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO));

  THCudaIntTensor_free(state, row_idx);
  THCudaIntTensor_free(state, col_idx);
  THCudaDoubleTensor_free(state, val);

  return 0;
}


int spadd_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaDoubleTensor *A_val,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaDoubleTensor *B_val,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *C_val,
    const double alpha, const double beta, const long rows, const long cols) {

  long nnzA = THCudaDoubleTensor_size(state, A_val, 0);
  long nnzB = THCudaDoubleTensor_size(state, B_val, 0);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  // Grab reference
  A_csr_row = THCudaIntTensor_newContiguous(state, A_csr_row);
  A_csr_col = THCudaIntTensor_newContiguous(state, A_csr_col);
  A_val = THCudaDoubleTensor_newContiguous(state, A_val);
  B_csr_row = THCudaIntTensor_newContiguous(state, B_csr_row);
  B_csr_col = THCudaIntTensor_newContiguous(state, B_csr_col);
  B_val = THCudaDoubleTensor_newContiguous(state, B_val);

  // Setup
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int *p_rowA = THCudaIntTensor_data(state, A_csr_row);
  int *p_colA = THCudaIntTensor_data(state, A_csr_col);
  double *p_valA = THCudaDoubleTensor_data(state, A_val);

  int *p_rowB = THCudaIntTensor_data(state, B_csr_row);
  int *p_colB = THCudaIntTensor_data(state, B_csr_col);
  double *p_valB = THCudaDoubleTensor_data(state, B_val);

  THCudaIntTensor_resize1d(state, C_csr_row, rows+1);
  int *p_rowC = THCudaIntTensor_data(state, C_csr_row);

  int nnzC;
  int* nnzTotalDevHostPtr = &nnzC;
  // nnzTotalDevHostPtr points to host memory
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  THCusparseCheck(cusparseXcsrgeamNnz(
      handle, rows, cols,
      descr, nnzA, p_rowA, p_colA,
      descr, nnzB, p_rowB, p_colB,
      descr, p_rowC, nnzTotalDevHostPtr));

  if(NULL != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    int baseC;
    THCudaCheck(
        cudaMemcpy(&nnzC, p_rowC+rows, sizeof(int), cudaMemcpyDeviceToHost));
    THCudaCheck(
        cudaMemcpy(&baseC, p_rowC, sizeof(int), cudaMemcpyDeviceToHost));
    nnzC -= baseC;
  }

  THCudaIntTensor_resize1d(state, C_csr_col, nnzC);
  THCudaDoubleTensor_resize1d(state, C_val, nnzC);
  int *p_colC = THCudaIntTensor_data(state, C_csr_col);
  double *p_valC = THCudaDoubleTensor_data(state, C_val);

  THCusparseCheck(cusparseDcsrgeam(
      handle, rows, cols,
      &alpha, descr, nnzA, p_valA, p_rowA, p_colA,
      &beta, descr, nnzB, p_valB, p_rowB, p_colB,
      descr, p_valC, p_rowC, p_colC));

  // Release references
  THCudaIntTensor_free(state, A_csr_row);
  THCudaIntTensor_free(state, A_csr_col);
  THCudaDoubleTensor_free(state, A_val);
  THCudaIntTensor_free(state, B_csr_row);
  THCudaIntTensor_free(state, B_csr_col);
  THCudaDoubleTensor_free(state, B_val);

  return 0;
}

int spadd_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaDoubleTensor *gradA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaDoubleTensor *gradB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *gradC,
    const double alpha, const double beta, const long rows, const long cols) {

  THCAssertSameGPU(THCudaDoubleTensor_checkGPU(
        state, 9,
        A_csr_row, A_csr_col, gradA,
        B_csr_row, B_csr_col, gradB,
        C_csr_row, C_csr_col, gradC));

  long nnzA = THCudaIntTensor_size(state, A_csr_col, 0);
  long nnzB = THCudaIntTensor_size(state, B_csr_col, 0);
  long nnzC = THCudaIntTensor_size(state, C_csr_col, 0);

  // Grab a reference
  A_csr_row = THCudaIntTensor_newContiguous(state, A_csr_row);
  A_csr_col = THCudaIntTensor_newContiguous(state, A_csr_col);
  B_csr_row = THCudaIntTensor_newContiguous(state, B_csr_row);
  B_csr_col = THCudaIntTensor_newContiguous(state, B_csr_col);
  C_csr_row = THCudaIntTensor_newContiguous(state, C_csr_row);
  C_csr_col = THCudaIntTensor_newContiguous(state, C_csr_col);
  gradC = THCudaDoubleTensor_newContiguous(state, gradC);

  // Prepare outputs
  THCudaDoubleTensor_resize1d(state, gradA, nnzA);
  THCudaDoubleTensor_resize1d(state, gradB, nnzB);
  THCudaDoubleTensor_zero(state, gradA);
  THCudaDoubleTensor_zero(state, gradB);

  int* p_csr_rowA = THCudaIntTensor_data(state, A_csr_row);
  int* p_csr_colA = THCudaIntTensor_data(state, A_csr_col);
  int* p_csr_rowB = THCudaIntTensor_data(state, B_csr_row);
  int* p_csr_colB = THCudaIntTensor_data(state, B_csr_col);
  int* p_csr_rowC = THCudaIntTensor_data(state, C_csr_row);
  int* p_csr_colC = THCudaIntTensor_data(state, C_csr_col);
  double* p_gradA = THCudaDoubleTensor_data(state, gradA);
  double* p_gradB = THCudaDoubleTensor_data(state, gradB);
  double* p_gradC = THCudaDoubleTensor_data(state, gradC);

  // Setup cusparse
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int* p_coo_rowC;
  THCudaCheck(THCudaMalloc(state, (void**) &p_coo_rowC, nnzC*sizeof(int)));
  THCusparseCheck(cusparseXcsr2coo(
      handle, p_csr_rowC, nnzC, rows, p_coo_rowC, CUSPARSE_INDEX_BASE_ZERO));

  spadd_backward_cuda(
      p_csr_rowA, p_csr_colA, p_gradA, nnzA,
      p_csr_rowB, p_csr_colB, p_gradB, nnzB,
      p_coo_rowC, p_csr_colC, p_gradC, nnzC,
      alpha, beta, rows, cols);

  // Free scratch data
  THCudaCheck(THCudaFree(state, p_coo_rowC));

  // Release references
  THCudaIntTensor_free(state, A_csr_row);
  THCudaIntTensor_free(state, A_csr_col);
  THCudaIntTensor_free(state, B_csr_row);
  THCudaIntTensor_free(state, B_csr_col);
  THCudaIntTensor_free(state, C_csr_row);
  THCudaIntTensor_free(state, C_csr_col);
  THCudaDoubleTensor_free(state, gradC);
  return 0;
}


int spmv(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col, THCudaDoubleTensor *val,
    THCudaDoubleTensor *vector,
    THCudaDoubleTensor *output,
    const long rows, const long cols, const int transpose) {

  THCAssertSameGPU(THCudaDoubleTensor_checkGPU(
        state, 5, csr_row, csr_col, val, vector, output));

  long nnz = THCudaDoubleTensor_size(state, val, 0);

  THArgCheck(rows+1 == THCudaIntTensor_size(state, csr_row, 0), 0,
      "csr rows should have rows+1 entries");
  THArgCheck(nnz == THCudaIntTensor_size(state, csr_col, 0), 1,
      "csr cols should have nnz entries");

  int vector_size = THCudaDoubleTensor_size(state, vector, 0);
  if(transpose == 1) {
    THArgCheck(rows == vector_size, 3,
               "rows should match vector size in transpose"
               "mode got %d expected %d", rows, vector_size);
    THCudaDoubleTensor_resize1d(state, output, cols);
  } else {
    THArgCheck(cols == vector_size,
        3, "cols should match vector size in non-transpose mode");
    THCudaDoubleTensor_resize1d(state, output, rows);
  }
  THCudaDoubleTensor_zero(state, output);

  // Grab a reference
  csr_row = THCudaIntTensor_newContiguous(state, csr_row);
  csr_col = THCudaIntTensor_newContiguous(state, csr_col);
  val = THCudaDoubleTensor_newContiguous(state, val);
  vector = THCudaDoubleTensor_newContiguous(state, vector);

  // Setup cusparse
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int *p_row = THCudaIntTensor_data(state, csr_row);
  int *p_col = THCudaIntTensor_data(state, csr_col);
  double *p_val = THCudaDoubleTensor_data(state, val);
  double *p_vector = THCudaDoubleTensor_data(state, vector);
  double *p_output = THCudaDoubleTensor_data(state, output);

  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  if(transpose == 1) {
    trans = CUSPARSE_OPERATION_TRANSPOSE;
  }

  /* TODO(mgharbi): more accurate version when transposing: */
  /* convert to CSC and run with NON_TRANSPOSE. */
  double multiplier = 1.0f;
  THCusparseCheck(cusparseDcsrmv(handle, trans,
        rows, cols, nnz, &multiplier, descr, p_val, p_row, p_col,
        p_vector, &multiplier, p_output));

  // Release references
  THCudaIntTensor_free(state, csr_row);
  THCudaIntTensor_free(state, csr_col);
  THCudaDoubleTensor_free(state, val);
  THCudaDoubleTensor_free(state, vector);
  return 0;
}


int spmv_backward_matrix(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col,
    THCudaDoubleTensor *vector,
    THCudaDoubleTensor *grad_output, THCudaDoubleTensor *grad_matrix,
    const long rows, const long cols) {

  // C = AB
  // dL/dA = dL/dC.Bt
  // dL/dB = At.dL/dC

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  long nnz = THCudaIntTensor_size(state, csr_col, 0);
  int *p_csrRow = THCudaIntTensor_data(state, csr_row);
  int *p_csrCol = THCudaIntTensor_data(state, csr_col);

  // Grab references
  csr_row = THCudaIntTensor_newContiguous(state, csr_row);
  csr_col = THCudaIntTensor_newContiguous(state, csr_col);
  vector = THCudaDoubleTensor_newContiguous(state, vector);
  grad_output = THCudaDoubleTensor_newContiguous(state, grad_output);

  int* p_cooRow;
  THCudaCheck(THCudaMalloc(state, (void**) &p_cooRow, nnz*sizeof(int)));
  THCusparseCheck(cusparseXcsr2coo(
      handle, p_csrRow, nnz, rows, p_cooRow, CUSPARSE_INDEX_BASE_ZERO));

  THCudaDoubleTensor_resize1d(state, grad_matrix, nnz);
  THCudaDoubleTensor_zero(state, grad_matrix);

  double* p_vector = THCudaDoubleTensor_data(state, vector);
  double* p_grad_output = THCudaDoubleTensor_data(state, grad_output);
  double* p_grad_matrix = THCudaDoubleTensor_data(state, grad_matrix);
  spmv_backward_matrix_cuda(
      p_cooRow, p_csrCol, p_vector, p_grad_output, p_grad_matrix,
      rows, cols, nnz);

  // Free scratch data
  THCudaCheck(THCudaFree(state, p_cooRow));

  // Release references
  THCudaIntTensor_free(state, csr_row);
  THCudaIntTensor_free(state, csr_col);
  THCudaDoubleTensor_free(state, vector);
  THCudaDoubleTensor_free(state, grad_output);

  return 0;
}


int spmm_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaDoubleTensor *A_val,
    const long rowsA, const long colsA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaDoubleTensor *B_val,
    const long rowsB, const long colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *C_val) {

  THAssertMsg(colsA == rowsB, "spmm: A and B should have"
              " compatible inner dimensions.");
  long nnzA = THCudaDoubleTensor_size(state, A_val, 0);
  long nnzB = THCudaDoubleTensor_size(state, B_val, 0);

  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  // Grab reference
  A_csr_row = THCudaIntTensor_newContiguous(state, A_csr_row);
  A_csr_col = THCudaIntTensor_newContiguous(state, A_csr_col);
  A_val = THCudaDoubleTensor_newContiguous(state, A_val);
  B_csr_row = THCudaIntTensor_newContiguous(state, B_csr_row);
  B_csr_col = THCudaIntTensor_newContiguous(state, B_csr_col);
  B_val = THCudaDoubleTensor_newContiguous(state, B_val);

  // Setup
  cusparseMatDescr_t descr=0;
  THCusparseCheck(cusparseCreateMatDescr(&descr));
  THCusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  THCusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

  int *p_rowA = THCudaIntTensor_data(state, A_csr_row);
  int *p_colA = THCudaIntTensor_data(state, A_csr_col);
  double *p_valA = THCudaDoubleTensor_data(state, A_val);

  int *p_rowB = THCudaIntTensor_data(state, B_csr_row);
  int *p_colB = THCudaIntTensor_data(state, B_csr_col);
  double *p_valB = THCudaDoubleTensor_data(state, B_val);

  THCudaIntTensor_resize1d(state, C_csr_row, rowsA+1);
  int *p_rowC = THCudaIntTensor_data(state, C_csr_row);

  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  int nnzC;
  int* nnzTotalDevHostPtr = &nnzC;
  // nnzTotalDevHostPtr points to host memory
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  THCusparseCheck(cusparseXcsrgemmNnz(
      handle, transA, transB, rowsA, rowsB, colsA,
      descr, nnzA, p_rowA, p_colA,
      descr, nnzB, p_rowB, p_colB,
      descr, p_rowC, nnzTotalDevHostPtr));

  if(NULL != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    int baseC;
    THCudaCheck(cudaMemcpy(
          &nnzC, p_rowC+rowsA, sizeof(int), cudaMemcpyDeviceToHost));
    THCudaCheck(cudaMemcpy(
          &baseC, p_rowC, sizeof(int), cudaMemcpyDeviceToHost));
    nnzC -= baseC;
  }

  THCudaIntTensor_resize1d(state, C_csr_col, nnzC);
  THCudaDoubleTensor_resize1d(state, C_val, nnzC);
  int *p_colC = THCudaIntTensor_data(state, C_csr_col);
  double *p_valC = THCudaDoubleTensor_data(state, C_val);

  THCusparseCheck(cusparseDcsrgemm(
      handle, transA, transB, rowsA, colsB, colsA,
      descr, nnzA, p_valA, p_rowA, p_colA,
      descr, nnzB, p_valB, p_rowB, p_colB,
      descr, p_valC, p_rowC, p_colC));

  // Release references
  THCudaIntTensor_free(state, A_csr_row);
  THCudaIntTensor_free(state, A_csr_col);
  THCudaDoubleTensor_free(state, A_val);
  THCudaIntTensor_free(state, B_csr_row);
  THCudaIntTensor_free(state, B_csr_col);
  THCudaDoubleTensor_free(state, B_val);

  return 0;
}


int spmm_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col,
    THCudaDoubleTensor *A_val, THCudaDoubleTensor *A_grad_val,
    const long rowsA, const long colsA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col,
    THCudaDoubleTensor *B_val, THCudaDoubleTensor *B_grad_val,
    const long rowsB, const long colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *C_grad_val) {

  THAssertMsg(colsA == rowsB, "spmm: A and B should have"
              " compatible inner dimensions.");

  long nnzA = THCudaIntTensor_size(state, A_csr_col, 0);
  long nnzB = THCudaIntTensor_size(state, B_csr_col, 0);
  long nnzC = THCudaIntTensor_size(state, C_csr_col, 0);

  // Grab reference
  A_csr_row = THCudaIntTensor_newContiguous(state, A_csr_row);
  A_csr_col = THCudaIntTensor_newContiguous(state, A_csr_col);
  A_val = THCudaDoubleTensor_newContiguous(state, A_val);
  B_csr_row = THCudaIntTensor_newContiguous(state, B_csr_row);
  B_csr_col = THCudaIntTensor_newContiguous(state, B_csr_col);
  B_val = THCudaDoubleTensor_newContiguous(state, B_val);
  C_csr_row = THCudaIntTensor_newContiguous(state, C_csr_row);
  C_csr_col = THCudaIntTensor_newContiguous(state, C_csr_col);
  C_grad_val = THCudaDoubleTensor_newContiguous(state, C_grad_val);

  THCudaDoubleTensor_resize1d(state, A_grad_val, nnzA);
  THCudaDoubleTensor_resize1d(state, B_grad_val, nnzB);
  THCudaDoubleTensor_zero(state, A_grad_val);
  THCudaDoubleTensor_zero(state, B_grad_val);

  int *p_rowA = THCudaIntTensor_data(state, A_csr_row);
  int *p_colA = THCudaIntTensor_data(state, A_csr_col);

  int *p_rowB = THCudaIntTensor_data(state, B_csr_row);
  int *p_colB = THCudaIntTensor_data(state, B_csr_col);

  int *p_rowC = THCudaIntTensor_data(state, C_csr_row);
  int *p_colC = THCudaIntTensor_data(state, C_csr_col);
  double *p_grad_valC = THCudaDoubleTensor_data(state, C_grad_val);

  // Setup cusparse
  cusparseHandle_t handle = THCState_getCurrentSparseHandle(state);

  { // dL/dA = dL/dC.Bt
    double *p_grad_valA = THCudaDoubleTensor_data(state, A_grad_val);
    double *p_valB = THCudaDoubleTensor_data(state, B_val);
    int* p_coo_rowA;
    THCudaCheck(THCudaMalloc(state, (void**) &p_coo_rowA, nnzA*sizeof(int)));
    THCusparseCheck(cusparseXcsr2coo(
        handle, p_rowA, nnzA, rowsA, p_coo_rowA, CUSPARSE_INDEX_BASE_ZERO));
    matmul_preserve_sparsity_cuda(
        p_rowC, p_colC, p_grad_valC,
        p_rowB, p_colB, p_valB,
        p_coo_rowA, p_colA, p_grad_valA, nnzA);
    THCudaCheck(THCudaFree(state, p_coo_rowA));
  }

  { // dL/dB = At.dL/dC
    double *p_grad_valB = THCudaDoubleTensor_data(state, B_grad_val);
    double *p_valA = THCudaDoubleTensor_data(state, A_val);

    int* p_coo_rowB;
    THCudaCheck(THCudaMalloc(state, (void**) &p_coo_rowB, nnzB*sizeof(int)));
    THCusparseCheck(cusparseXcsr2coo(
        handle, p_rowB, nnzB, rowsB, p_coo_rowB, CUSPARSE_INDEX_BASE_ZERO));

    // transpose A
    int* p_csc_rowA;
    THCudaCheck(THCudaMalloc(state, (void**) &p_csc_rowA, nnzA*sizeof(int)));
    int* p_csc_colA;
    THCudaCheck(THCudaMalloc(state, (void**) &p_csc_colA, (colsA+1)*sizeof(int)));
    double* p_csc_valA;
    THCudaCheck(THCudaMalloc(state, (void**) &p_csc_valA, nnzA*sizeof(double)));
    THCusparseCheck(cusparseDcsr2csc(
        handle, rowsA, colsA, nnzA,
        p_valA, p_rowA, p_colA,
        p_csc_valA, p_csc_rowA, p_csc_colA,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO));

    // transpose dL/dC
    int colsC = colsB;
    int rowsC = rowsA;
    int* p_csc_rowC;
    THCudaCheck(THCudaMalloc(state, (void**) &p_csc_rowC, nnzC*sizeof(int)));
    int* p_csc_colC;
    THCudaCheck(THCudaMalloc(state, (void**) &p_csc_colC, (colsC+1)*sizeof(int)));
    double* p_csc_grad_valC;
    THCudaCheck(THCudaMalloc(state, (void**) &p_csc_grad_valC, nnzC*sizeof(double)));
    THCusparseCheck(cusparseDcsr2csc(
        handle, rowsC, colsC, nnzC,
        p_grad_valC, p_rowC, p_colC,
        p_csc_grad_valC, p_csc_rowC, p_csc_colC,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO));

    matmul_preserve_sparsity_cuda(
        p_csc_colA, p_csc_rowA, p_csc_valA,
        p_csc_colC, p_csc_rowC, p_csc_grad_valC,
        p_coo_rowB, p_colB, p_grad_valB, nnzB);

    THCudaCheck(THCudaFree(state, p_coo_rowB));
    THCudaCheck(THCudaFree(state, p_csc_rowA));
    THCudaCheck(THCudaFree(state, p_csc_colA));
    THCudaCheck(THCudaFree(state, p_csc_valA));
    THCudaCheck(THCudaFree(state, p_csc_rowC));
    THCudaCheck(THCudaFree(state, p_csc_colC));
    THCudaCheck(THCudaFree(state, p_csc_grad_valC));
  }

  // Release references
  THCudaIntTensor_free(state, A_csr_row);
  THCudaIntTensor_free(state, A_csr_col);
  THCudaDoubleTensor_free(state, A_val);
  THCudaIntTensor_free(state, B_csr_row);
  THCudaIntTensor_free(state, B_csr_col);
  THCudaDoubleTensor_free(state, B_val);
  THCudaIntTensor_free(state, C_csr_row);
  THCudaIntTensor_free(state, C_csr_col);
  THCudaDoubleTensor_free(state, C_grad_val);

  return 0;
}
