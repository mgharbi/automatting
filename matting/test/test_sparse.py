import numpy as np
import torch as th
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import profiler

import matting.sparse as sp
import matting.optim as optim
import matting.functions.sparse as spfuncs


def _get_random_sparse_matrix(nrows, ncols, nnz):
  row = np.random.randint(0, nrows, size=(nnz,), dtype=np.int32)
  col = np.random.randint(0, ncols, size=(nnz,), dtype=np.int32)

  tuples = [(a, b) for a, b in zip(row, col)]
  unique_tuples = sorted(set(tuples)) #, key=lambda x: tuples.index(x))
  row, col = zip(*unique_tuples)
  row = np.array(row)
  col = np.array(col)
  nnz = row.size

  row = th.from_numpy(row).cuda()
  col = th.from_numpy(col).cuda()
  val = th.from_numpy(np.random.uniform(size=(nnz,)).astype(np.float32)).cuda()
  A = sp.from_coo(row, col, val, th.Size((nrows, ncols)))
  return A

def test_permutation():
  for i in range(10):
    nrows = 10
    ncols = 11
    nnz = 9
    row = np.random.randint(0, nrows, size=(nnz,), dtype=np.int32)
    col = np.random.randint(0, ncols, size=(nnz,), dtype=np.int32)

    tuples = [(a, b) for a, b in zip(row, col)]
    unique_tuples = set(tuples) #, key=lambda x: tuples.index(x))
    # unique_tuples = sorted(set(tuples)) #, key=lambda x: tuples.index(x))
    row, col = zip(*unique_tuples)
    row = np.array(row)
    col = np.array(col)
    nnz = row.size

    row = th.from_numpy(row).cuda()
    col = th.from_numpy(col).cuda()
    val = th.from_numpy(np.random.uniform(size=(nnz,)).astype(np.float32)).cuda()
    A = sp.from_coo(row, col, val, th.Size((nrows, ncols)))

    gradcheck(spfuncs.Coo2Csr.apply,
        (A.csr_row_idx.data, A.col_idx.data, A.val.data, A.size),
        eps=1e-4, atol=1e-5, rtol=1e-3,
        raise_exception=True)

def test_transpose():
  np.random.seed(0)
  for i in range(10):
    nrows = np.random.randint(3,8)
    ncols = np.random.randint(3,8)
    nnz = np.random.randint(1,nrows*ncols/2)
    A = _get_random_sparse_matrix(nrows, ncols, nnz)

    # A.make_variable()

    Ad = A.to_dense()
    At = sp.transpose(A)
    Atd = At.to_dense()

    diff = np.amax(np.abs(Ad.T-Atd))

    assert diff < 1e-5

    gradcheck(spfuncs.Transpose.apply,
        (A.csr_row_idx.data, A.col_idx.data, A.val.data, A.size),
        eps=1e-4, atol=1e-5, rtol=1e-3,
        raise_exception=True)


def test_spadd_gradients():
  np.random.seed(0)

  for i in range(10):
    nrows = np.random.randint(3,8)
    ncols = np.random.randint(3,8)
    nnz = np.random.randint(1,nrows*ncols/2)
    A = _get_random_sparse_matrix(nrows, ncols, nnz)
    B = _get_random_sparse_matrix(nrows, ncols, nnz)

    gradcheck(spfuncs.SpAdd.apply,
        (A.csr_row_idx, A.col_idx, A.val,
         B.csr_row_idx, B.col_idx, B.val,
         A.size, 1.0, 1.0), eps=1e-4, atol=1e-5, rtol=1e-3,
         raise_exception=True)


def test_spmv_gradients():
  np.random.seed(0)

  for i in range(10):
    nrows = np.random.randint(3,10)
    ncols = np.random.randint(3,10)
    nnz = np.random.randint(1,nrows*ncols/2)

    A = _get_random_sparse_matrix(nrows, ncols, nnz)
    vector = th.from_numpy(
        np.random.uniform(size=(ncols,)).astype(np.float32)).cuda()

    # A.make_variable()
    vector = Variable(vector, requires_grad=True)

    gradcheck(spfuncs.SpMV.apply,
        (A.csr_row_idx, A.col_idx, A.val,
         vector, A.size), eps=1e-3, atol=1e-4, rtol=1e-3,
         raise_exception=True)


def test_spmm_gradients():
  np.random.seed(0)

  i = 0
  while i < 10:
    nrows = np.random.randint(3,8)
    ncols = np.random.randint(3,8)
    ncols2 = np.random.randint(3,8)
    nnz = np.random.randint(1,nrows*ncols/2)
    nnz2 = np.random.randint(1,ncols*ncols2/2)
    A = _get_random_sparse_matrix(nrows, ncols, nnz)
    B = _get_random_sparse_matrix(ncols, ncols2, nnz2)

    C = sp.spmm(A, B)
    if C.nnz == 0: # TODO: handle this special case for backprop
      continue
    i += 1

    gradcheck(spfuncs.SpMM.apply,
        (A.csr_row_idx, A.col_idx, A.val, A.size,
         B.csr_row_idx, B.col_idx, B.val, B.size),
         eps=1e-4, atol=2e-4, rtol=1e-3,
         raise_exception=True)


def test_coo2csr():
  row = th.from_numpy(np.array(
        [0, 0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 3, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  val = th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda()
  n = 4
  A = sp.from_coo(row, col, val, th.Size((n, n)))

  csr_row_idx = A.csr_row_idx.data.cpu().numpy()
  col_idx = A.col_idx.data.cpu().numpy()
  val2 = A.val.data.cpu().numpy()

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
  val = Variable(th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  n = 4
  A = sp.from_coo(row, col, val, th.Size((n, n)))
  B = sp.from_coo(row, col, val, th.Size((n, n)))

  C = sp.spadd(A, B)
  assert (C.csr_row_idx.data.cpu().numpy() == A.csr_row_idx.data.cpu().numpy()).all()
  assert (C.col_idx.data.cpu().numpy() == A.col_idx.data.cpu().numpy()).all()
  assert (C.val.data.cpu().numpy() == np.array([0, 2, 4, 6])).all()


def test_add_different_sparsity():
  n = 4
  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  valA = Variable(th.from_numpy(np.ones(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  A = sp.from_coo(row, col, valA, th.Size((n, n)))

  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [1, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  valB = Variable(th.from_numpy(np.ones(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  B = sp.from_coo(row, col, valB, th.Size((n, n)))

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
  valA = Variable(th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  n = 4
  A = sp.from_coo(row, col, valA, th.Size((n, n+1)))

  v = Variable(th.ones(n+1).cuda(), requires_grad=True)
  
  out = sp.spmv(A, v)
  assert out.size()[0] == n
  assert np.amax(np.abs(out.data.cpu().numpy() - np.array([0, 1, 2, 3]))) < 1e-5

  loss = out.sum()
  loss.backward()

  assert np.amax(np.abs(v.grad.data.cpu().numpy() - np.array([0, 1, 2, 3, 0]))) < 1e-5
  assert np.amax(np.abs(valA.grad.data.cpu().numpy() - np.array([1, 1, 1, 1]))) < 1e-5


def test_multiply_same_sparsity():
  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  valA = Variable(th.from_numpy(np.arange(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  n = 4
  A = sp.from_coo(row, col, valA, th.Size((n, n)))

  valB = Variable(th.from_numpy(np.array([3, 6, 1, 8], dtype=np.float32)).cuda(), requires_grad=True)
  B = sp.from_coo(row, col, valB, th.Size((n, n)))
  
  C = sp.spmm(A, B)
  assert (C.val.data.cpu().numpy() == np.array([0, 6, 2, 24])).all()

  loss = C.val.sum()
  loss.backward()

  assert np.amax(np.abs(valB.grad.data.cpu().numpy() - np.array([0, 1, 2, 3]))) < 1e-5
  assert np.amax(np.abs(valA.grad.data.cpu().numpy() - np.array([3, 6, 1, 8]))) < 1e-5


def test_multiply_different_sparsity():
  n = 4
  row = th.from_numpy(np.array(
        [0, 0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [0, 3, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  valA = Variable(th.from_numpy(np.ones(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  A = sp.from_coo(row, col, valA, th.Size((n, n)))

  row = th.from_numpy(np.array(
        [0, 1, 2, 3], dtype=np.int32)).cuda()
  col = th.from_numpy(np.array(
        [1, 1, 2, 3], dtype=np.int32)).cuda()
  nnz = row.numel()
  valB = Variable(th.from_numpy(np.ones(nnz, dtype=np.float32)).cuda(), requires_grad=True)
  B = sp.from_coo(row, col, valB, th.Size((n, n)))

  # A.make_variable()
  # B.make_variable()
  
  C = sp.spmm(A, B)

  row_idx = C.csr_row_idx.data.cpu().numpy()
  col_idx = C.col_idx.data.cpu().numpy()
  valC = C.val

  assert (row_idx == np.array([0, 2, 3, 4, 5], dtype=np.int32)).all()
  assert (col_idx == np.array([1, 3, 1, 2, 3], dtype=np.int32)).all()
  assert (valC.data.cpu().numpy() == np.array([1, 1, 1, 1, 1])).all()

  loss = valC.sum()
  loss.backward()

  assert (valA.grad.numel() == A.nnz)
  assert (valB.grad.numel() == B.nnz)

def test_cg():
  nnz = 10
  np.random.seed(0)
  row = np.arange(nnz, dtype=np.int32)
  col = np.arange(nnz, dtype=np.int32)

  row = th.from_numpy(row).cuda()
  col = th.from_numpy(col).cuda()
  val = th.from_numpy(np.random.uniform(size=(nnz,)).astype(np.float32)).cuda()
  val = Variable(val, requires_grad=True)
  A = sp.from_coo(row, col, val, th.Size((nnz, nnz)))
  
  optimizer = th.optim.Adam([val], lr=1e-1)
  avg_loss = 0  # Running average of the loss for display
  for step in range(2000):
    b = Variable(th.from_numpy(np.random.uniform(size=(nnz,)).astype(np.float32)).cuda(), requires_grad=False)
    x0 = Variable(th.zeros(nnz).cuda(), requires_grad=False)

    x_opt, err = optim.sparse_cg(A,b, x0, steps=10)

    target = 0.5*b  # we want to learn 2*Identity matrix
    loss = (target-x_opt).pow(2).sum()
    avg_loss = 0.9*avg_loss + 0.1*loss

    if step % 100 == 0:
      msg = "Step {:5d} loss = {:8.5f} cg residual = {:4.2g}".format(step, avg_loss.data[0], err)
      print msg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  Ad = A.to_dense()
  diff = np.amax(np.abs(Ad-2*np.identity(nnz)))
  
  assert diff < 1e-3


def test_performance():
  nnz = 10000
  nrows = 100000
  ncols = 100000
  A = _get_random_sparse_matrix(nrows, ncols, nnz)
  B = _get_random_sparse_matrix(nrows, ncols, nnz)
  # A.make_variable()
  # B.make_variable()

  v = Variable(th.ones(ncols).cuda(), requires_grad=True)
  
  with profiler.profile() as prof:
    out = sp.spmv(A, v)
    out2 = sp.spmm(A, B)
    out3 = sp.spadd(A, B)

    loss = out.sum() + out2.val.sum() + out3.val.sum()
    loss.backward()
  print (prof)


# TODO(mgharbi):
# - m-m add: gradient w.r.t. scalar and gradients
# - argument checks
# - tranpose op
# - Special case of spmm when C.nnz == 0
# THCAssertSameGPU
