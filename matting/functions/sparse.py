import torch
from torch.autograd import Function
from .._ext import sparse

class SpMv(Function):
  """Sparse matrix-vector product"""

  def __init__(self, rows, cols):
    self.rows = rows
    self.cols = cols

  def forward(self, mtx_indices, mtx_values, vector):
    output = torch.FloatTensor()
    sparse.spmv_forward(
        mtx_indices, mtx_values, vector, output, self.rows, self.cols)
    return output

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
