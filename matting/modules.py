import copy
import logging
import sys
import time

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse as scisp
import skimage.io

import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matting.sparse as sp
import matting.optim as optim

from torchlib.modules import LinearChain

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MattingCNN(nn.Module):
  def __init__(self):
    super(MattingCNN, self).__init__()


    nn.LeakyReLU

    self.net = LinearChain(3, 4, width=64, depth=3, batchnorm=True)
    self.system = MattingSystem()
    self.solver = MattingSolver()

    # Learnable lmbda
    self.lmbda = nn.Parameter(th.ones(1)*100.0)

  def forward(self, sample):
    # TODO(mgharbi): these should be variables, and eventually a network's output
    assert sample['image'].shape[0] == 1  # NOTE: we do not handle batches at this point
    h = sample['image'].shape[2]
    w = sample['image'].shape[3]
    N = h*w
    weights = self.net(sample['image'])
    weights = weights.view(4, h*w)

    CM_weights  = weights[0, :]
    LOC_weights = weights[1, :]
    IU_weights  = weights[2, :]
    KU_weights  = weights[3, :]

    # cm_mult  = 1.0;
    # loc_mult = 1.0;
    # iu_mult  = 0.01;
    # ku_mult  = 0.05;
    # lmbda    = 100.0;
    # CM_weights  = Variable(cm_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())
    # LOC_weights = Variable(loc_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())
    # IU_weights  = Variable(iu_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())
    # KU_weights  = Variable(ku_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())

    single_sample = {}
    for k in sample.keys():
      if "Tensor" not in type(sample[k]).__name__:
        single_sample[k] = sample[k][0, ...]

    A, b = self.system(single_sample, CM_weights, LOC_weights, IU_weights, KU_weights,
                       self.lmbda, N)
    matte = self.solver(A, b)
    # matte = weights[0, :]
    matte = matte.view(1, h, w)
    # matte = np.clip(matte, 0, 1)
    return matte


class MattingSolver(nn.Module):
  def __init__(self, steps=50, verbose=True):
    self.steps = steps
    self.verbose = verbose
    super(MattingSolver, self).__init__()

  def forward(self, A, b):
    start = time.time()
    x0 = Variable(th.zeros(b.shape[0]).cuda(), requires_grad=False)
    x_opt, err = optim.sparse_cg(A, b, x0, steps=self.steps, verbose=self.verbose)
    end = time.time()
    if self.verbose:
      log.info("solve system {:.2f}s".format((end-start)))
    return x_opt


class MattingSystem(nn.Module):
  """docstring for MattingSystem"""
  def __init__(self):
    super(MattingSystem, self).__init__()
    
  def forward(self, sample, CM_weights, LOC_weights, IU_weights, KU_weights, lmbda, N):
    start = time.time()
    Lcm = self._color_mixture(N, sample, CM_weights)
    Lmat = self._matting_laplacian(N, sample, LOC_weights)
    Lcs = self._intra_unknowns(N, sample, IU_weights)

    kToUconf = sample['kToUconf']
    known = sample['known']
    kToU = sample['kToU']

    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
    linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

    KU = sp.Sparse(linear_csr_row_idx, linear_idx, KU_weights.mul(kToUconf), th.Size((N,N)))
    known = sp.Sparse(linear_csr_row_idx, linear_idx, lmbda*known, th.Size((N,N)))

    A = sp.spadd(Lcs, sp.spadd(Lmat, (sp.spadd(sp.spadd(Lcm, KU), known))))
    b = sp.spmv(sp.spadd(KU, known), kToU)
    end = time.time()
    log.info("prepare system {:.2f}s/im".format((end-start)))

    return A, b

  def _color_mixture(self, N, sample, CM_weights):
    # CM
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
    linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

    # This one reorders the data, but is is ok we dont differentiate through it
    Wcm = sp.from_coo(sample["Wcm_row"], sample["Wcm_col"].view(-1),
                      sample["Wcm_data"], th.Size((N, N)))

    # Below needs to be differentiable
    diag = sp.Sparse(linear_csr_row_idx, linear_idx, CM_weights, th.Size((N, N)))
    Wcm = sp.spmm(diag, Wcm)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wcm, ones)
    Wcm.mul_(-1.0)
    Lcm = sp.spadd(sp.from_coo(linear_idx, linear_idx, row_sum.data, th.Size((N, N))), Wcm)
    Lcmt = sp.transpose(Lcm)
    Lcm = sp.spmm(Lcmt, Lcm)
    return Lcm

  def _matting_laplacian(self, N, sample, LOC_weights):
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())

    w = sample['image'].shape[-1]
    h = sample['image'].shape[-2]

    # Matting Laplacian
    inInd = sample["LOC_inInd"]
    weights = LOC_weights[inInd.long().view(-1)]
    flows = sample['LOC_flows']
    flow_sz = flows.shape[0]
    tiled_weights = weights.view(1, 1, -1).repeat(flow_sz, flow_sz, 1)
    flows = flows.mul(tiled_weights)

    neighInds = th.cat(
        [inInd-1-w, inInd-1, inInd-1+w, inInd-w, inInd, inInd+w, inInd+1-w, inInd+1, inInd+1+w], 1)

    # Wmat = None
    for i in range(9):
      iRows = neighInds[:, i].clone().view(-1, 1).repeat(1, 9)
      iFlows = flows[:, i, :].permute(1, 0).clone()
      iWmat = sp.from_coo(iRows.view(-1), neighInds.view(-1), iFlows.data.view(-1), th.Size((N, N)))  # <--- this is not differentiable
      if i == 0:
        Wmat = iWmat
      else:
        Wmat = sp.spadd(Wmat, Wmat)
      break
    Wmatt = sp.transpose(Wmat)
    Wmat = sp.spadd(Wmat, Wmatt)
    Wmat.mul_(0.5)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wmat, ones)
    Wmat.mul_(-1.0)
    diag = sp.from_coo(linear_idx, linear_idx, row_sum, th.Size((N, N)))  # <------- this is not diff
    Lmat = sp.spadd(diag, Wmat)
    return Lmat

  def _intra_unknowns(self, N, sample, IU_weights):
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())

    weights = IU_weights[sample["IU_inInd"].long().view(-1)]
    nweights = weights.numel()
    flows = sample['IU_flows'][:, :5]  # NOTE(mgharbi): bug in the data, row6 shouldnt be used
    flow_sz = flows.shape[1]
    flows = flows.mul(weights.view(-1, 1).repeat(1, flow_sz))
    neighInd = sample["IU_neighInd"][:, :5].contiguous()
    inInd = sample["IU_inInd"].clone()
    inInd = inInd.repeat(1, neighInd.shape[1])
    Wcs = sp.from_coo(inInd.view(-1), neighInd.view(-1), flows.data.view(-1), th.Size((N, N)))
    Wcst = sp.transpose(Wcs)
    Wcs = sp.spadd(Wcs, Wcst)
    Wcs.mul_(0.5)
    ones = Variable(th.ones(N).cuda())
    row_sum = sp.spmv(Wcs, ones)
    Wcs.mul_(-1)
    diag = sp.from_coo(linear_idx, linear_idx, row_sum.data, th.Size((N, N)))
    Lcs = sp.spadd(diag, Wcs)
    return Lcs


def get(params):
  params = copy.deepcopy(params)  # do not touch the original
  model_name = params.pop("model", None)
  return getattr(sys.modules[__name__], model_name)(**params)
