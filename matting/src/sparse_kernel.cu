#include "sparse_kernel.h"
void spmv_backward_matrix_cuda(
    const int* p_cooRow, const int* p_csrCol, const float* p_vector,
    const float* p_grad_output, float* p_grad_matrix,
    const int rows, const int cols) {
}
