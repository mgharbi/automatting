# from .functions.sparse import

class SparseCOO(object):
  """"""
  def __init__(self, indices, values, size):
    self.indices = indices
    self.values = values
    self.size = size


def spadd(s_matA, s_matB):
  """Sum of sparse matrices"""
  # TODO
  return s_matA


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
