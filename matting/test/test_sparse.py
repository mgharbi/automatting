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

def test_spadd():
  n = 10
  A_rows = th.cat([th.LongTensor(range(n))]*2).view(1, -1)
  A_cols = th.cat([th.zeros(n).type(th.LongTensor), th.LongTensor(range(n))]).view(1, -1)
  A_indices = Variable(th.cat([A_rows, A_cols], 0).cuda())
  A_values  = Variable(th.FloatTensor(range(2*n)).cuda())

  B_indices = A_indices
  B_values  = A_values

  A = sp.SparseCOO(A_indices, A_values, th.Size((n, n)))
  B = sp.SparseCOO(B_indices, B_values, th.Size((n, n)))

  out = sp.spadd(A, B)
  print(out.indices.cpu())
  print(out.values.cpu())
  print(out.size)

