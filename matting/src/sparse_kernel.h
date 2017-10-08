#ifndef SPARSE_KERNEL_H_FTUJX27W
#define SPARSE_KERNEL_H_FTUJX27W

#ifdef __cplusplus
extern "C" {
#endif

void spmv_backward_matrix_cuda(
    const int* p_cooRow, const int* p_csrCol, const float* p_vector,
    const float* p_grad_output, float* p_grad_matrix,
    const int rows, const int cols);

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: SPARSE_KERNEL_H_FTUJX27W */
