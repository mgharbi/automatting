// int spmv_forward(
//   THLongTensor *mtx_indices, THFloatTensor *mtx_values, 
//   THFloatTensor *vector, THFloatTensor *output, const int rows, const int cols);

// int sample_weighting_backward(
//     THFloatTensor *grad_output,
//     THFloatTensor *grad_samples,
//     THFloatTensor *grad_params,
//     THFloatTensor *grad_weights);

// int spdiag_mm_forward(
//   THFloatTensor *diagonal,
//   THLongTensor *mtx_indices, THFloatTensor *mtx_values, 
//   THFloatTensor *output, const int rows, const int cols);

int coo2csr(THCudaIntTensor *row_idx, 
            THCudaIntTensor *col_idx,
            THCudaTensor *val,
            THCudaIntTensor *csr_row_idx,
            const int rows, const int cols);

int csr2csc(THCudaIntTensor *row_idx, 
            THCudaIntTensor *col_idx,
            THCudaTensor *val,
            THCudaIntTensor *csc_row_idx, 
            THCudaIntTensor *csc_col_idx,
            THCudaTensor *csc_val,
            const int rows, const int cols);

int spadd_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *A_val,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *B_val,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_val,
    const float alpha, const float beta, const int rows, const int cols);

int spadd_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *gradA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *gradB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *gradC,
    const float alpha, const float beta, const int rows, const int cols);


int spmv(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col, THCudaTensor *val,
    THCudaTensor *vector,
    THCudaTensor *output,
    const int rows, const int cols, const int transpose);


int spmv_backward_matrix(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col,
    THCudaTensor *vector,
    THCudaTensor *grad_output,
    THCudaTensor *grad_matrix,
    const int rows, const int cols);

int spmm_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaTensor *A_val,
    const int rowsA, const int colsA, 
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaTensor *B_val,
    const int rowsB, const int colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_val);

int spmm_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col,
    THCudaTensor *A_val, THCudaTensor *A_grad_val,
    const int rowsA, const int colsA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col,
    THCudaTensor *B_val, THCudaTensor *B_grad_val,
    const int rowsB, const int colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaTensor *C_grad_val);

