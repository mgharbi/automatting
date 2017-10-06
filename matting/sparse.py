import matting.functions.sparse as spfuncs

class SparseCOO(object):
  """"""
  def __init__(self, indices, values, size):
    self.indices = indices
    self.values = values
    self.size = size


def spadd(A, B):
  """Sum of sparse matrices"""
  op = spfuncs.SpAdd(A.size[0], A.size[1])
  idx, val = op(A.indices, A.values, B.indices, B.values)
  return SparseCOO(idx, val, A.size)


def sp_gram(s_mat):
  """A^T.A for A sparse"""
  pass


def sp_laplacian(s_mat):
  """diag(row_sum(A)) - A for A sparse"""
  pass


def spmv(s_matA, v):
  """Sparse matrix - dense vector product."""
  pass


def spdsm(s_diag, s_mat):
  """Sparse diagonal - sparse matrix product."""
  pass
