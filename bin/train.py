#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging

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
import matting.modules as modules

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def make_variable(d, cuda=True):
  ret = {}
  for k in d.keys():
    if "Tensor" not in type(d[k]).__name__:
      ret[k] = d[k]
      continue
    if cuda:
      ret[k] = Variable(d[k].cuda())
    else:
      ret[k] = Variable(d[k])
  return ret


def main(args):
  dataset = dset.MattingDataset(
      args.dataset, transform=dset.ToTensor())

  for sample_idx, sample in enumerate(dataset):
    sample = make_variable(sample, cuda=True)

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

    system = modules.MattingSystem()
    A, b = system(sample, CM_weights, LOC_weights, IU_weights, KU_weights, lmbda)

    start = time.time()
    x0 = Variable(th.zeros(N).cuda(), requires_grad=False)
    x_opt, err = optim.sparse_cg(A, b, x0, steps=10, verbose=False)
    end = time.time()
    log.info("solve system {:.2f}s".format((end-start)))

    matte = x_opt.data.cpu().numpy()
    matte = np.reshape(matte, [h, w])
    matte = np.clip(matte, 0, 1)

    # skimage.io.imsave("torch_matte{}.png".format(sample_idx), matte)


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
