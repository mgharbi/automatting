import torch
from torch.autograd import Function
from .._ext import sparse

def coo2csr(row_idx, col_idx, val, size):
  csr_row_idx = torch.IntTensor().cuda()
  sparse.coo2csr(row_idx, col_idx, val, csr_row_idx, size[0], size[1])
  return (csr_row_idx, col_idx, val)

class SpAdd(Function):
  """Sum of sparse matrices."""

  @staticmethod
  def forward(ctx, rowA, colA, valA, rowB, colB, valB, size):
    ctx.save_for_backward(rowA, colA, rowB, colB, size)
    rowC = torch.IntTensor().cuda()
    colC = torch.IntTensor().cuda()
    valC = torch.FloatTensor().cuda()
    sparse.spadd_forward(
        rowA, colA, valA, 
        rowB, colB, valB, 
        rowC, colC, valC, 
        size[0], size[1])
    return rowC, colC, valC

  @staticmethod
  def backward(ctx, grad_output):
    rowA, colA, rowB, colB, size = ctx.saved_variables
    grad_rowA = grad_colA = grad_valA = None
    grad_rowB = grad_colB = grad_valB = None
    grad_size = None

    # dL/dA_ik = dL/dC_ik*dC_ik/dA_ik where
    # dC_ik/dA_ik = 1 iff A_ik != 0
    # dL/dA_ik should select in dL/dC_ik following A's sparsity pattern
    # TODO

    return grad_rowA, grad_colA, grad_valA, \
           grad_rowB, grad_colB, grad_valB, grad_size
