#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse as scisp
import skimage.io

import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import time

import matting.dataset as dset
import matting.sparse as sp
import matting.optim as optim

def main(args):
  dataset = dset.MattingDataset(
      args.dataset, transform=dset.ToTensor())

  start = time.time()
  sample = dataset[0]
  end = time.time()
  print("load sample {:.2f}s/im".format((end-start)))

  h = sample['height']
  w = sample['width']
  N = h*w

  # TODO(mgharbi): these should be variables, and eventually a network's output
  cm_mult  = 1.0;
  loc_mult = 1.0;
  iu_mult  = 0.01;
  ku_mult  = 0.05;
  lmbda    = 100.0;
  CM_weights  = Variable(cm_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())
  LOC_weights = Variable(loc_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())
  IU_weights  = Variable(iu_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())
  KU_weights  = Variable(ku_mult*th.from_numpy(np.ones((N,), dtype=np.float32)).cuda())

  start = time.time()

  # CM
  linear_idx = th.from_numpy(np.arange(N, dtype=np.int32)).cuda()
  linear_csr_row_idx = th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda()
  Wcm = sp.from_coo(sample["Wcm_row"].cuda(), sample["Wcm_col"].cuda(), sample["Wcm_data"].cuda(), th.Size((N, N)))
  diag = sp.Sparse(linear_csr_row_idx, linear_idx, CM_weights, th.Size((N, N)))
  Wcm = sp.spmm(diag, Wcm)
  row_sum = sp.spmv(Wcm, th.ones(N).cuda())
  Wcm.mul_(-1.0)
  Lcm = sp.spadd(sp.from_coo(linear_idx, linear_idx, row_sum.data, th.Size((N, N))), Wcm)
  Lcmt = sp.transpose(Lcm)
  Lcm = sp.spmm(Lcmt, Lcm)

  # # IU
  # weights = IU_weights[sample["IU_inInd"].long().cuda().view(-1)]
  # nweights = weights.numel()
  # flows = Variable(sample['IU_flows'].cuda())
  # flow_sz = flows.shape[1]
  # flows = flows.mul(weights.view(-1, 1).repeat(1, flow_sz))
  # neighInd = sample["IU_neighInd"].cuda()
  # inInd = sample["IU_inInd"].clone()
  # inInd = inInd.repeat(1, neighInd.shape[1]).cuda()
  # # TODO: proper ordering? differentiable coo2csr?
  # fake_data = np.zeros(flows.view(-1).shape)
  # fake = scisp.csr_matrix(
  #     (fake_data, (inInd.view(-1).cpu().numpy(), neighInd.view(-1).cpu().numpy())),
  #     shape=(N, N))
  # row_ptr = th.from_numpy(fake.indptr).cuda()
  # cols = th.from_numpy(fake.indices).cuda()
  # Wcs = sp.Sparse(row_ptr, cols, flows.view(-1), th.Size((N, N)))
  # # Wcs = sp.from_coo(inInd.view(-1), neighInd.view(-1), flows.view(-1), th.Size((N, N)))
  # # TODO: duplicate entries?
  # # TODO: ---------------
  # Wcs.make_variable()
  # Wcst = sp.transpose(Wcs)
  # Wcs = sp.spadd(Wcs, Wcst)
  # Wcs.mul_(0.5)
  # ones = Variable(th.ones(N).cuda())
  # row_sum = sp.spmv(Wcs, ones)
  # Wcs.mul_(-1)
  # Lcs = sp.spadd(sp.from_coo(linear_idx, linear_idx, row_sum, th.Size((N, N))), Wcs)

  kToUconf = Variable(sample['kToUconf'].cuda())
  known = Variable(sample['known'].cuda())
  kToU = Variable(sample['kToU'].cuda())

  KU = sp.Sparse(linear_csr_row_idx, linear_idx, KU_weights.mul(kToUconf), th.Size((N,N)))
  known = sp.Sparse(linear_csr_row_idx, linear_idx, lmbda*known, th.Size((N,N)))

  A = sp.spadd(sp.spadd(Lcm, KU), known)
  b = sp.spmv(sp.spadd(KU, known), kToU)

  end = time.time()
  print("prepare system {:.2f}s/im".format((end-start)))

  start = time.time()
  x0 = Variable(th.zeros(N).cuda(), requires_grad=False)
  x_opt, err = optim.sparse_cg(A, b, x0, steps=500, verbose=True)
  end = time.time()
  print("solve system {:.2f}s".format((end-start)))

  matte = x_opt.data.cpu().numpy()
  matte = np.reshape(matte, [h, w])
  matte = np.clip(matte, 0, 1)

  skimage.io.imsave("torch_matte.png", matte)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str)
  # parser.add_argument("output", type=str)
  parser.add_argument("--checkpoint", type=str)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--log_step", type=int, default=100)
  parser.add_argument("--checkpoint_step", type=int, default=10000)
  parser.add_argument("--visualization_step", type=int, default=10000)
  parser.add_argument("--nepochs", type=int, default=1)
  args = parser.parse_args()
  main(args)
