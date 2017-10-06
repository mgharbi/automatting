#include <THC/THC.h>
#include <stdio.h>

extern THCState *state;

int spadd_forward(
    THCudaLongTensor *A_idx, THCudaTensor *A_val,
    THCudaLongTensor *B_idx, THCudaTensor *B_val,
    THCudaLongTensor *out_idx, THCudaTensor *out_val,
    int rows, int cols) {

  THCudaTensor_resize1d(state, out_val, 10);
  THCudaLongTensor_resize2d(state, out_idx, 2, 10);
  THCudaTensor_zero(state, out_val);
  THCudaLongTensor_zero(state, out_idx);

  return 0;
}
