#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time
import datetime
import logging

import torch as th
from torch.autograd import Variable

import matting.optim as optim


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(args):
  npixels = 10

  # The stuff we want to learn
  A = Variable(th.randn(npixels, npixels).type(th.FloatTensor), requires_grad=True)

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
      Asym = A.transpose(1, 0).matmul(A)  # Symmetrize A
      x_0 = Variable(th.zeros(npixels).type(th.FloatTensor), requires_grad=False)
      x_opt, residual = optim.cg(Asym, b, x_0, steps=args.cg_max_step)

      target = 0.5*b  # we want to learn 2*Identity matrix
      loss = loss + (target-x_opt).pow(2).sum()

    avg_loss = 0.9*avg_loss + 0.1*loss

    if step % 100 == 0:
      elapsed = datetime.timedelta(seconds=time.time()-start_time)
      msg = "Step {:5d} [{}] loss = {:8.5f} cg residual = {:4.2g}".format(step, elapsed, avg_loss.data[0], residual)
      if step > 0:
        msg += " ({:.0f} samples/s)".format(step*args.batch_size/elapsed.seconds)
      log.info(msg)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  Asym = A.transpose(1, 0).matmul(A)
  test_x = Variable(th.ones(npixels), requires_grad=False)
  log.info("A:", Asym)
  log.info("A.ones (should be close to all 2s):", Asym.matmul(test_x))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--learning_rate", type=float, default=1e-2)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--max_step", type=int, default=2000)
  parser.add_argument("--cg_max_step", type=int, default=2)
  args = parser.parse_args()
  main(args)
