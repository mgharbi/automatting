import torch
from torch.autograd import Function
from .._ext import sparse

def coo2csr(row_idx, col_idx, val, size):
  csr_row_idx = torch.IntTensor().cuda()
  sparse.coo2csr(row_idx, col_idx, val, csr_row_idx, size[0], size[1])
  return (csr_row_idx, col_idx, val)

class SpAdd(Function):
  """Sum of sparse matrices"""

  @staticmethod
  def forward(ctx, rowA, colA, valA, rowB, colB, valB, size):
    rowC = torch.IntTensor().cuda()
    colC = torch.IntTensor().cuda()
    valC = torch.FloatTensor().cuda()
    sparse.spadd_forward(
        rowA, colA, valA, 
        rowB, colB, valB, 
        rowC, colC, valC, 
        size[0], size[1])
    return rowC, colC, valC

  # def backward(self, grad_output):
  #   grad_samples = torch.FloatTensor()
  #   grad_params = torch.FloatTensor()
  #   grad_weights = torch.FloatTensor()
  #   montecarlo_lib.sample_weighting_backward(
  #       grad_output, grad_samples, grad_params, grad_weights)
  #   return grad_samples, grad_params, grad_weights

# def spmv(mtx_indices, mtx_values, vector, rows, cols):
#   functor = SpMv(rows, cols)
#   return functor(mtx_indices, mtx_values, vector)
