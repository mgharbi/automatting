import torch
from torch.autograd import Function
from .._ext import sparse

class SpAdd(Function):
  """Sum of sparse matrices"""

  def __init__(self, rows, cols):
    self.rows = rows
    self.cols = cols

  def forward(self, A_idx, A_val, B_idx, B_val):
    out_idx = torch.LongTensor().cuda()
    out_val = torch.FloatTensor().cuda()
    sparse.spadd_forward(
        A_idx, A_val, 
        B_idx, B_val, 
        out_idx, out_val, 
        self.rows, self.cols)
    return out_idx, out_val

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
