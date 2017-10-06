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

int spadd_forward(
    THCudaLongTensor *A_idx, THCudaTensor *A_val,
    THCudaLongTensor *B_idx, THCudaTensor *B_val,
    THCudaLongTensor *out_idx, THCudaTensor *out_val,
    const int rows, const int cols);
