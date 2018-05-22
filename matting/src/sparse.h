int coo2csr(THCudaIntTensor *row_idx,
            THCudaIntTensor *col_idx,
            THCudaDoubleTensor *val,
            THCudaIntTensor *csr_row_idx,
            THCudaIntTensor *csr_col_idx,
            THCudaDoubleTensor *csr_val,
            THCudaIntTensor *permutation,
            const long rows, const long cols);

int csr2csc(THCudaIntTensor *row_idx,
            THCudaIntTensor *col_idx,
            THCudaDoubleTensor *val,
            THCudaIntTensor *csc_row_idx,
            THCudaIntTensor *csc_col_idx,
            THCudaDoubleTensor *csc_val,
            const long rows, const long cols);

int spadd_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaDoubleTensor *A_val,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaDoubleTensor *B_val,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *C_val,
    const double alpha, const double beta, const long rows, const long cols);

int spadd_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaDoubleTensor *gradA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaDoubleTensor *gradB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *gradC,
    const double alpha, const double beta, const long rows, const long cols);


int spmv(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col, THCudaDoubleTensor *val,
    THCudaDoubleTensor *vector,
    THCudaDoubleTensor *output,
    const long rows, const long cols, const int transpose);


int spmv_backward_matrix(
    THCudaIntTensor *csr_row, THCudaIntTensor *csr_col,
    THCudaDoubleTensor *vector,
    THCudaDoubleTensor *grad_output,
    THCudaDoubleTensor *grad_matrix,
    const long rows, const long cols);

int spmm_forward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col, THCudaDoubleTensor *A_val,
    const long rowsA, const long colsA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col, THCudaDoubleTensor *B_val,
    const long rowsB, const long colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *C_val);

int spmm_backward(
    THCudaIntTensor *A_csr_row, THCudaIntTensor *A_csr_col,
    THCudaDoubleTensor *A_val, THCudaDoubleTensor *A_grad_val,
    const long rowsA, const long colsA,
    THCudaIntTensor *B_csr_row, THCudaIntTensor *B_csr_col,
    THCudaDoubleTensor *B_val, THCudaDoubleTensor *B_grad_val,
    const long rowsB, const long colsB,
    THCudaIntTensor *C_csr_row, THCudaIntTensor *C_csr_col, THCudaDoubleTensor *C_grad_val);
