import numpy as np
import torch as th
from torch.autograd import Variable

import matting.sparse as sp

# def test_forward():
#   n = 10
#   mtx_rows = th.cat([th.LongTensor(range(n))]*2).view(1, -1)
#   mtx_cols = th.cat([th.zeros(n).type(th.LongTensor), th.LongTensor(range(n))]).view(1, -1)
#   mtx_indices = Variable(th.cat([mtx_rows, mtx_cols], 0))
#   mtx_values  = Variable(th.FloatTensor(range(2*n)))
#   vector = Variable(th.ones(n))
#
#   out = spmv(mtx_indices, mtx_values, vector, n, n)
#   print(out)

def test_coo2csr():
  row = th.from_numpy(np.array(
        [0, 0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 3, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda()
  n = 4
  A = sp.from_coo(row, col, val, th.Size((n, n)))

  csr_row_idx = A.csr_row_idx.cpu().numpy()
  col_idx = A.col_idx.cpu().numpy()
  val2 = A.val.cpu().numpy()

  assert csr_row_idx.size == n + 1
  assert (csr_row_idx == np.array([0, 2, 3, 4, 5], dtype=np.int32)).all()
  assert (col_idx == np.array([0, 3, 1, 2, 3], dtype=np.int32)).all()
  assert (val2 == val.cpu().numpy()).all()
  

def test_add_same_sparsity():
  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda()
  n = 4
  A = sp.from_coo(row, col, val, th.Size((n, n)))
  B = sp.from_coo(row, col, val, th.Size((n, n)))

  A.val = Variable(A.val)
  B.val = Variable(B.val)
  
  C = sp.spadd(A, B)
  assert (C.val.data.cpu().numpy() == np.array([0, 2, 4, 6])).all()


def test_add_different_sparsity():
  n = 4
  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.ones(nnz, dtype=np.float32)).cuda()
  A = sp.from_coo(row, col, val, th.Size((n, n)))

  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [1, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.ones(nnz, dtype=np.float32)).cuda()
  B = sp.from_coo(row, col, val, th.Size((n, n)))

  A.val = Variable(A.val)
  B.val = Variable(B.val)
  
  C = sp.spadd(A, B)

  row_idx = C.csr_row_idx.data.cpu().numpy()
  col_idx = C.col_idx.data.cpu().numpy()
  # val = C.val.data.cpu().numpy()
  assert (row_idx == np.array([0, 2, 3, 4, 5], dtype=np.int32)).all()
  assert (col_idx == np.array([0, 1, 1, 2, 3], dtype=np.int32)).all()
  assert (C.val.data.cpu().numpy() == np.array([1, 1, 2, 2, 2])).all()
