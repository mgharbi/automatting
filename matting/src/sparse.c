#include <THC/THC.h>
#include <stdio.h>

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

  if( THCudaIntTensor_nDimension(state, row_idx) != 1) {
    THError("row_idx should be 1D");
    return 1;
  }
  if( THCudaIntTensor_nDimension(state, col_idx) != 1) {
    THError("col_idx should be 1D");
    return 1;
  }
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
    const int rows, const int cols) {

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

  float multiplier = 1.0f;
  THCusparseCheck(cusparseScsrgeam(
      handle, rows, cols,
      &multiplier, descr, nnzA, p_valA, p_rowA, p_colA,
      &multiplier, descr, nnzB, p_valB, p_rowB, p_colB,
      descr, p_valC, p_rowC, p_colC));

  return 0;
}


int spmv_forward(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col, THCudaTensor *val,
    THCudaTensor *vector,
    THCudaTensor *output,
    const int rows, const int cols) {

  int nnz = THCudaTensor_size(state, val, 0);

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

  // Assert cols == vec size

  THCudaTensor_resize1d(state, output, cols);
  THCudaTensor_zero(state, output);
  float *p_output = THCudaTensor_data(state, output);

  float multiplier = 1.0f;
  THCusparseCheck(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        rows, cols, nnz, &multiplier, descr, p_val, p_row, p_col,
        p_vector, &multiplier, p_output));

  return 0;
}
