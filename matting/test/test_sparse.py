import torch as th
from torch.autograd import Variable

from ..functions.sparse import spmv

def test_forward():
  n = 10
  mtx_rows = th.cat([th.LongTensor(range(n))]*2).view(1, -1)
  mtx_cols = th.cat([th.zeros(n).type(th.LongTensor), th.LongTensor(range(n))]).view(1, -1)
  mtx_indices = Variable(th.cat([mtx_rows, mtx_cols], 0))
  mtx_values  = Variable(th.FloatTensor(range(2*n)))
  vector = Variable(th.ones(n))

  out = spmv(mtx_indices, mtx_values, vector, n, n)
  print(out)

