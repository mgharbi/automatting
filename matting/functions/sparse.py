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
  def forward(ctx, rowA, colA, valA, rowB, colB, valB, size, alpha=1.0, beta=1.0):
    ctx.matrix_size = size
    ctx.alpha = alpha
    ctx.beta = beta

    rowC = torch.IntTensor().cuda()
    colC = torch.IntTensor().cuda()
    valC = torch.FloatTensor().cuda()
    sparse.spadd_forward(
        rowA, colA, valA, 
        rowB, colB, valB, 
        rowC, colC, valC, 
        alpha, beta,
        size[0], size[1])

    ctx.save_for_backward(rowA, colA, rowB, colB, rowC, colC)
    return rowC, colC, valC

  @staticmethod
  def backward(ctx, grad_rowC, grad_colC, grad_valC):
    rowA, colA, rowB, colB, rowC, colC = ctx.saved_variables
    size = ctx.matrix_size
    alpha = ctx.alpha
    beta = ctx.beta

    grad_rowA = grad_colA = grad_valA = None
    grad_rowB = grad_colB = grad_valB = None
    grad_size = None
    grad_alpha = None
    grad_beta = None

    # dL/dA_ik = dL/dC_ik*dC_ik/dA_ik where
    # dC_ik/dA_ik = 1 iff A_ik != 0
    # dL/dA_ik should select in dL/dC_ik following A's sparsity pattern
    grad_valA = grad_valC.data.new()
    grad_valB = grad_valC.data.new()
    sparse.spadd_backward(
        rowA.data, colA.data, grad_valA,
        rowB.data, colB.data, grad_valB,
        rowC.data, colC.data, grad_valC.data,
        alpha, beta, size[0], size[1])

    grad_valA = Variable(grad_valA)
    grad_valB = Variable(grad_valB)

    return grad_rowA, grad_colA, grad_valA, \
           grad_rowB, grad_colB, grad_valB, grad_size, grad_alpha, grad_beta


class SpMV(Function):
  """Sparse matrix-vector product."""

  @staticmethod
  def forward(ctx, row, col, val, vector, size):
    ctx.save_for_backward(row, col, val, vector)
    output = vector.new() 
    sparse.spmv(
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
    sparse.spmv(
        row.data, col.data, val.data, 
        grad_output.data, grad_vector,
        nrows, ncols, True)

    grad_val = val.data.new()
    sparse.spmv_backward_matrix(
        row.data, col.data, 
        vector.data, grad_output.data, grad_val,
        nrows, ncols)

    grad_vector = Variable(grad_vector)
    grad_val = Variable(grad_val)

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
        rowA, colA, valA, sizeA[0], sizeA[1], False,
        rowB, colB, valB, sizeB[0], sizeB[1], False,
        rowC, colC, valC)
    return rowC, colC, valC
