#include <TH/TH.h>
#include <cstdio>

extern "C"{
  int spmv_forward(THLongTensor *mtx_indices, THFloatTensor
      *mtx_values, THFloatTensor *vector, THFloatTensor *output, 
      const int rows, const int cols) { 
    if (THLongTensor_nDimension(mtx_indices) != 2) {
      printf("indices should be 2D\n");
      return 0;
    }
    if (THFloatTensor_nDimension(mtx_values) != 1)  {
      printf("values should be 1D\n");
      return 0; 
    }
    if (THFloatTensor_nDimension(vector) != 1){
      printf("vector should be 1D\n");
      return 0;
    }
    if (THLongTensor_size(mtx_indices, 0) != 2) {
      printf("indices should have dim0 = 2\n");
      return 0;
    }
    int64_t vec_cols = THFloatTensor_size(vector, 0);
    if (vec_cols != cols) {
      printf("vector should have %lld columns, got %d} = 2\n", vec_cols, cols);
      return 0;
    }

    int n_indices = THLongTensor_size(mtx_indices, 1);

    THFloatTensor_resize1d(output, rows);
    THFloatTensor_zero(output);

    const int64_t *idx    = THLongTensor_data(mtx_indices);
    const float *mtx_data = THFloatTensor_data(mtx_values);
    const float *v_data   = THFloatTensor_data(vector);
    float *out      = THFloatTensor_data(output);

    for (int i = 0; i < n_indices; ++i) {
      const int64_t row = idx[i];
      const int64_t col = idx[i + n_indices];
      out[row] += mtx_data[i]*v_data[col];
    }
    return 1;
  }
}
