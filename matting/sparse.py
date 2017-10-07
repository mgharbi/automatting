import torch as th
import matting.functions.sparse as spfuncs

def from_coo(row_idx, col_idx, val, size):
  """Construct a sparse matrix from THTensors describing a COO format."""
  csr_row_idx, col_idx, val = spfuncs.coo2csr(row_idx, col_idx, val, size)
  return Sparse(csr_row_idx, col_idx, val, size)


class Sparse(object):
  """"""
  def __init__(self, csr_row_idx, col_idx, val, size):
    self.csr_row_idx = csr_row_idx
    self.col_idx = col_idx
    self.val = val
    self.size = size
    self.storage = "csr"

  def __str__(self):
    s = "Sparse matrix {}\n".format(self.size)
    s += "  csr_row {}\n".format(self.csr_row_idx)
    s += "  col {}\n".format(self.col_idx)
    s += "  val {}\n".format(self.val)
    return s


def spadd(A, B):
  """Sum of sparse matrices"""
  rowC, colC, valC = spfuncs.SpAdd.apply(
      A.csr_row_idx, A.col_idx, A.val,
      B.csr_row_idx, B.col_idx, B.val,
      A.size)
  return Sparse(rowC, colC, valC, A.size)


def sp_gram(s_mat):
  """A^T.A for A sparse"""
  pass


def sp_laplacian(s_mat):
  """diag(row_sum(A)) - A for A sparse"""
  pass


def spmv(A, v):
  """Sparse matrix - dense vector product."""
  return spfuncs.SpMv.apply(A.csr_row_idx, A.col_idx, A.val, v, A.size)


def spdsm(s_diag, s_mat):
  """Sparse diagonal - sparse matrix product."""
  pass
