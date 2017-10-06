#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import datetime
import logging


import torch as th
from torch.autograd import Variable

from matting.functions.sparse import spmv
import matting.optim as optim

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(args):
  npixels = 10
  diag_idx = th.LongTensor(range(npixels)).repeat(2, 1)
  diag = 2*th.sparse.FloatTensor(diag_idx, th.ones(npixels), th.Size((npixels, npixels)))
  A = Variable(th.FloatTensor(th.randn(1)))*diag.mul(diag)

  import ipdb; ipdb.set_trace()
  return
  # A = diag.mul(Variable(th.FloatTensor(1), requires_grad=True))

  optimizer = th.optim.Adam([A], lr=args.learning_rate)
  avg_loss = 0  # Running average of the loss for display
  start_time = time.time()
  for step in range(args.max_step):
    # Average loss over a mini-batch
    loss = 0
    for example in range(args.batch_size):
      # Sample some data
      b = Variable(th.randn(npixels).type(th.FloatTensor), requires_grad=False)

      # Forward pass
      x_0 = Variable(th.zeros(npixels).type(th.FloatTensor), requires_grad=False)
      x_opt, residual = optim.sparse_cg(A, b, x_0, steps=args.cg_max_step)

      target = 0.5*b  # we want to learn 2*Identity matrix
      loss = loss + (target-x_opt).pow(2).sum()

    avg_loss = 0.9*avg_loss + 0.1*loss

    if step % 100 == 0:
      elapsed = datetime.timedelta(seconds=time.time()-start_time)
      msg = "Step {:5d} [{}] loss = {:8.5f} cg residual = {:4.2g}".format(
          step, elapsed, avg_loss.data[0], residual)
      # if step > 0:
        # msg += " ({:.0f} samples/s)".format(step*args.batch_size/elapsed.seconds)
      log.info(msg)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    # optimizer.step()

    test_x = Variable(th.ones(npixels), requires_grad=False)
    # log.info("A:", A)
    # log.info("A.ones (should be close to all 2s): {}".format(spmv(A, test_x)))

  # vector = Variable(th.ones(npixels).type(th.FloatTensor))
  # out = spmv(diag, vector)
  #
  # x_0 = Variable(th.zeros(npixels).type(th.FloatTensor), requires_grad=False)
  # x_opt, residual = optim.sparse_cg(diag, vector, x_0, steps=1)

  # print(x_opt, vector)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--learning_rate", type=float, default=1e-2)
  parser.add_argument("--batch_size", type=int, default=11)
  parser.add_argument("--max_step", type=int, default=200)
  parser.add_argument("--cg_max_step", type=int, default=2)
  args = parser.parse_args()
  main(args)
