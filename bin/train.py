#!/usr/bin/env python

import argparse
import logging
import os
import setproctitle
import time

import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

import matting.dataset as dataset
import matting.modules as modules

import torchlib.viz as viz
from torchlib.utils import save
from torchlib.utils import make_variable
from torchlib.image import crop_like

log = logging.getLogger(__name__)


PROCESS_NAME = "automatting"


def main(args, params):
  data = dataset.MattingDataset(args.data_dir, transform=dataset.ToTensor())
  val_data = dataset.MattingDataset(args.data_dir, transform=dataset.ToTensor())

  dataloader = DataLoader(data, 
      batch_size=1,
      shuffle=False, num_workers=0)

  val_dataloader = DataLoader(val_data, 
      batch_size=min(len(val_data), 4), shuffle=False, num_workers=0)

  log.info("Training with {} samples".format(len(data)))

  # Starting checkpoint file
  checkpoint = os.path.join(args.output, "checkpoint.ph")
  if args.checkpoint is not None:
    checkpoint = args.checkpoint

  chkpt = None
  if os.path.isfile(checkpoint):
    log.info("Resuming from checkpoint {}".format(checkpoint))
    chkpt = th.load(checkpoint)
    params = chkpt['params']  # override params

  log.info("Model parameters: {}".format(params))

  model = modules.get(params)

  loss_fn = th.nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  global_step = 0

  if chkpt is not None:
    model.load_state_dict(chkpt['model_state'])
    optimizer.load_state_dict(chkpt['optimizer'])
    global_step = chkpt['step']

  # Destination checkpoint file
  checkpoint = os.path.join(args.output, "checkpoint.ph")

  name = os.path.basename(args.output)
  loss_viz = viz.ScalarVisualizer("loss", env=name)
  image_viz = viz.BatchVisualizer("images", env=name)
  matte_viz = viz.BatchVisualizer("mattes", env=name)

  log.info("Model: {}\n".format(model))

  model.cuda()
  loss_fn.cuda()

  log.info("Starting training from step {}".format(global_step))

  smooth_loss = 0
  smooth_time = 0
  ema_alpha = 0.99
  last_checkpoint_time = time.time()
  try:
    epoch = 0
    while True:
      # Train for one epoch
      for step, batch in enumerate(dataloader):
        batch_start = time.time()
        frac_epoch =  epoch+1.0*step/len(dataloader)

        batch_v = make_variable(batch, cuda=True)

        optimizer.zero_grad()
        output = model(batch_v)
        target = crop_like(batch_v['matte'], output)
        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()
        global_step += 1

        batch_end = time.time()
        smooth_loss = ema_alpha*loss.data[0] + (1.0-ema_alpha)*smooth_loss
        smooth_time = ema_alpha*(batch_end-batch_start) + (1.0-ema_alpha)*smooth_time

        if global_step % args.log_step == 0:
          log.info("Epoch {:.1f} | loss = {:.7f} | {:.1f} samples/s".format(
            frac_epoch, smooth_loss, target.shape[0]/smooth_time))

        if args.viz_step > 0 and global_step % args.viz_step == 0:
          model.train(False)
          for val_batch in val_dataloader:
            val_batchv = make_variable(val_batch, cuda=True)
            output = model(val_batchv)
            target = crop_like(val_batchv['matte'], output)
            val_loss = loss_fn(output, target)

            mini, maxi = target.min(), target.max()

            diff = (th.abs(output-target))
            vizdata = th.cat((target, output, diff), 0)
            vizdata = (vizdata-mini)/(maxi-mini)
            imgs = np.power(np.clip(vizdata.cpu().data, 0, 1), 1.0/2.2)
            imgs = np.expand_dims(imgs, 1)

            image_viz.update(val_batchv['image'].cpu().data, per_row=1)
            matte_viz.update(
                imgs,
                caption="Epoch {:.1f} | mse = {:.6f} | target, output, diff".format(
                  frac_epoch, val_loss.data[0]), per_row=3)
            log.info("  viz at step {}, loss = {:.6f}".format(global_step, val_loss.cpu().data[0]))
            break  # Only one batch for validation

          loss_viz.update(frac_epoch, smooth_loss)

          model.train(True)

        if batch_end-last_checkpoint_time > args.checkpoint_interval:
          last_checkpoint_time = time.time()
          save(checkpoint, model, params, optimizer, global_step)


      epoch += 1
      if args.epochs > 0 and epoch >= args.epochs:
        log.info("Ending training at epoch {} of {}".format(epoch, args.epochs))
        break

  except KeyboardInterrupt:
    log.info("training interrupted at step {}".format(global_step))
    checkpoint = os.path.join(args.output, "on_stop.ph")
    save(checkpoint, model, params, optimizer, global_step)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir')
  parser.add_argument('output')
  parser.add_argument('--val_data_dir')
  parser.add_argument('--checkpoint')
  parser.add_argument('--epochs', type=int, default=-1)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0)
  parser.add_argument('--debug', dest="debug", action="store_true")
  parser.add_argument('--params', nargs="*", default=["model=MattingCNN"])

  parser.add_argument('--log_step', type=int, default=25)
  parser.add_argument('--checkpoint_interval', type=int, default=1200, help='in seconds')
  parser.add_argument('--viz_step', type=int, default=5000)
  parser.set_defaults(debug=False)
  args = parser.parse_args()

  params = {}
  if args.params is not None:
    for p in args.params:
      k, v = p.split("=")
      if v.isdigit():
        v = int(v)
      params[k] = v

  logging.basicConfig(
      format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
  if args.debug:
    log.setLevel(logging.DEBUG)
  else:
    log.setLevel(logging.INFO)
  setproctitle.setproctitle('{}_{}'.format(PROCESS_NAME, os.path.basename(args.output)))


  main(args, params)