import torch
from torch.autograd import Function
from torch.autograd import Variable
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


class SpMV(Function):
  """Sparse matrix-vector product."""

  @staticmethod
  def forward(ctx, row, col, val, vector, size):
    ctx.save_for_backward(row, col, val, vector)
    output = vector.new() 
    sparse.spmv_forward(
        row, col, val, 
        vector, output,
        size[0], size[1], False)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    row, col, val, vector = ctx.saved_variables
    grad_row = grad_col = grad_val = None
    grad_vector = None
    grad_size = None

    nrows = row.data.shape[0]-1
    ncols = vector.data.shape[0]

    grad_vector = vector.data.new()

    sparse.spmv_forward(
        row.data, col.data, val.data, 
        grad_output.data, grad_vector,
        nrows, ncols, True)

    grad_vector = Variable(grad_vector)

    grad_val = val

    return grad_row, grad_col, grad_val, grad_vector, grad_size


class SpMM(Function):
  """Product of sparse matrices."""

  @staticmethod
  def forward(ctx, rowA, colA, valA, sizeA, rowB, colB, valB, sizeB):
    # ctx.save_for_backward(rowA, colA, rowB, colB, size)
    rowC = torch.IntTensor().cuda()
    colC = torch.IntTensor().cuda()
    valC = torch.FloatTensor().cuda()
    sparse.spmm_forward(
        rowA, colA, valA, sizeA[0], sizeA[1],
        rowB, colB, valB, sizeB[0], sizeB[1],
        rowC, colC, valC)
    return rowC, colC, valC
