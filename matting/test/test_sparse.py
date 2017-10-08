import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
# from torch.autograd import profiler

import matting.sparse as sp


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
  assert (C.csr_row_idx.data.cpu().numpy() == A.csr_row_idx.cpu()).all()
  assert (C.col_idx.data.cpu().numpy() == A.col_idx.cpu()).all()
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

  assert (row_idx == np.array([0, 2, 3, 4, 5], dtype=np.int32)).all()
  assert (col_idx == np.array([0, 1, 1, 2, 3], dtype=np.int32)).all()
  assert (C.val.data.cpu().numpy() == np.array([1, 1, 2, 2, 2])).all()


def test_matrix_vector():
  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda()
  n = 4
  A = sp.from_coo(row, col, val, th.Size((n, n+1)))

  A.val = Variable(A.val, requires_grad=True)
  A.csr_row_idx = Variable(A.csr_row_idx)
  A.col_idx = Variable(A.col_idx)
  v = Variable(th.ones(n+1).cuda(), requires_grad=True)
  
  out = sp.spmv(A, v)
  assert out.size()[0] == n
  assert np.amax(np.abs(out.data.cpu().numpy() - np.array([0, 1, 2, 3]))) < 1e-5

  loss = out.sum()
  loss.backward()

  assert np.amax(np.abs(v.grad.data.cpu().numpy() - np.array([0, 1, 2, 3, 0]))) < 1e-5
  assert np.amax(np.abs(A.val.grad.data.cpu().numpy() - np.array([1, 1, 1, 1]))) < 1e-5

  gradcheck(sp.spmv, (A, v), eps=1e-4, atol=1e-6, raise_exception=True)


def test_multiply_same_sparsity():
  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda()
  n = 4
  A = sp.from_coo(row, col, val, th.Size((n, n)))

  val = th.from_numpy(np.array([3, 6, 1, 8], dtype=np.float32)).cuda()
  B = sp.from_coo(row, col, val, th.Size((n, n)))

  A.val = Variable(A.val)
  B.val = Variable(B.val)
  
  C = sp.spmm(A, B)
  assert (C.val.data.cpu().numpy() == np.array([0, 6, 2, 24])).all()


def test_multiply_different_sparsity():
  n = 4
  row = th.from_numpy(np.array(
        [0, 0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 3, 1, 2, 3], dtype=np.int32)).cuda()
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
  
  C = sp.spmm(A, B)

  row_idx = C.csr_row_idx.data.cpu().numpy()
  col_idx = C.col_idx.data.cpu().numpy()
  val = C.val.data.cpu().numpy()

  assert (row_idx == np.array([0, 2, 3, 4, 5], dtype=np.int32)).all()
  assert (col_idx == np.array([1, 3, 1, 2, 3], dtype=np.int32)).all()
  assert (C.val.data.cpu().numpy() == np.array([1, 1, 1, 1, 1])).all()

# TODO(mgharbi):
# - mv gradient w.r.t to matrix
# - m-m gradient w.r.t to both matrices
# - m-m add with scalar and gradients
# - proper handling of transpose and sizes
# - argument checks
# THCAssertSameGPU

