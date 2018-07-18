"""
Opimise weights for alpha computation in image matting in Sparse way
Author: Shu Liu
Email: liush@ethz.ch
devel log: v21c
Description: use CG to compute sparse Lcm, Lmat, Luu from flow *.mat files
"""
from __future__ import print_function

import copy
import logging
import sys
import time

import numpy as np
import scipy.io
import scipy.sparse as scisp
import scipy.ndimage.filters as filters
import skimage.io

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import torchvision.transforms as transforms

import matting.sparse as sp

from torchlib.modules import LinearChain
from torchlib.modules import SkipAutoencoder
from torchlib.image import crop_like

import scipy.sparse as ssp

import os
import random
from random import randint
import string
import datetime
from os import listdir
from os.path import isfile, join
import re
import copy
import scipy.io as spio
import imageio
from numpy import linalg as LA
import matting.optim as optim
import matting.modules as mmo
import matting.dataset_v2 as dataset

log = logging.getLogger(__name__)

th.set_default_tensor_type('torch.DoubleTensor')

dataloader = dataset.MattingDataset('Training')

lmbda = Variable(th.from_numpy(np.array([100.0], dtype=np.float64)).cuda(),
                 requires_grad=False)

matting_sys = mmo.MattingSystem()
matting_loc = mmo.MattingSystem_LOC()


def validation_v3(sample, x0, alpha, param = None, valid_steps = 100, thresh = 1e-16):
    N = sample['image'].shape[0] * sample['image'].shape[1]

    if param is None:
        LOC_weights = Variable(th.from_numpy(np.ones(N).astype(np.float64)).cuda(), requires_grad=False)
    else:
        LOC_weights = Variable(th.from_numpy(param).cuda(), requires_grad=False)

    CM_weights = Variable(th.from_numpy(np.ones(N).astype(np.float64)).cuda(), requires_grad=False)
    IU_weights = Variable(th.from_numpy(np.ones(N).astype(np.float64)*0.01).cuda(), requires_grad=False)
    KU_weights = Variable(th.from_numpy(np.ones(N).astype(np.float64)*0.05).cuda(), requires_grad=False)

    Lmat_sp =  matting_sys._matting_laplacian(N = N, sample = sample, LOC_weights = LOC_weights)

    kToUconf = sample['kToUconf'].view(-1)
    known = sample['known'].view(-1)
    kToU = sample['kToU'].view(-1)
    linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
    linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

    KU = sp.Sparse(linear_csr_row_idx, linear_idx, KU_weights.mul(kToUconf), th.Size((N,N)))
    known = sp.Sparse(linear_csr_row_idx, linear_idx, lmbda.mul(known), th.Size((N,N)))

    Asys = sp.spadd(Lmat_sp, sp.spadd(KU, known))
    # could be changed to Asys = sp.spadd(Lcm_sp (or Luu_sp), sp.spadd(KU, known)) to compute weights of Lcm or Luu
    b = sp.spmv(sp.spadd(KU, known), kToU)

    res, _, _ = optim.sparse_cg_ib2(Asys, b, x0, steps = valid_steps, thresh = thresh)

    alpha = np.reshape(sample['matte'], [-1], 'F')
    alpha = Variable(th.from_numpy(alpha.astype(np.float64)).cuda(), requires_grad=False)

    tmp_err = np.mean((res.cpu().data.numpy()-alpha.cpu().data.numpy())**2)
    print('validation loss:', tmp_err)

    np.save('debug_DataLoader', res.cpu().data.numpy())
    return res, tmp_err


class Net_v02(th.nn.Module):
    def __init__(self, init_LOC_weights):
        super(Net_v02, self).__init__()

        self.LOC_weights = Variable(th.from_numpy(init_LOC_weights).cuda(), requires_grad=True)
        self.N = len(init_LOC_weights)

        # self.CM_weights = Variable(th.from_numpy(np.ones(self.N).astype(np.float64)).cuda(), requires_grad=False)
        # self.IU_weights = Variable(th.from_numpy(np.ones(self.N).astype(np.float64)*0.01).cuda(), requires_grad=False)
        self.KU_weights = Variable(th.from_numpy(np.ones(self.N).astype(np.float64)*0.05).cuda(), requires_grad=False)


    def forward(self, sample, x0, steps, N):

        lmbda = Variable(th.from_numpy(np.array([100.0], dtype=np.float64)).cuda(),
                         requires_grad=False)

        Lmat_sp =  matting_sys._matting_laplacian(N = self.N, sample = sample, LOC_weights = self.LOC_weights)

        kToUconf = sample['kToUconf'].view(-1)
        known = sample['known'].view(-1)
        kToU = sample['kToU'].view(-1)
        linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
        linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

        KU = sp.Sparse(linear_csr_row_idx, linear_idx, self.KU_weights.mul(kToUconf), th.Size((N,N)))
        known = sp.Sparse(linear_csr_row_idx, linear_idx, lmbda.mul(known), th.Size((N,N)))
        Asys = sp.spadd(Lmat_sp, sp.spadd(KU, known))
        b = sp.spmv(sp.spadd(KU, known), kToU)

        x0, _, _ = optim.sparse_cg_ib2(Asys, b, x0, steps = steps, thresh = 1e-16)

        return x0

for file_id in range(2, dataloader.__len__()):
    print('File %02d' % (file_id + 1) )

    tmp_dir_name = 'param_for_learn_v2/res_' + ('GT%02d' % (file_id+1))
    if not os.path.exists(tmp_dir_name):
        os.makedirs(tmp_dir_name)


    sample = dataloader.__getitem__(file_id)
    alpha = np.reshape(sample['matte'], [-1], 'F')
    alpha = Variable(th.from_numpy(alpha.astype(np.float64)).cuda(), requires_grad=False)

    N = sample['image'].shape[0] * sample['image'].shape[1]

    x0 = Variable(th.from_numpy(np.zeros(N).astype(np.float64)).cuda(), requires_grad=False)

    init_LOC_weights_np  = np.ones(N).astype(np.float64)

    aux_x0 = []
    aux_err = []
    for _ in range(5):
        pred, err = validation_v3(sample, x0, alpha, param = init_LOC_weights_np, valid_steps = 400, thresh = 1e-16)
        x0 = Variable(th.from_numpy(pred.cpu().data.numpy().astype(np.float64)).cuda(), requires_grad=False)
        aux_x0.append(x0.cpu().data.numpy().astype(np.float64))
        aux_err.append(err)

    smallest_ind = np.argmin(aux_err)
    x0 = Variable(th.from_numpy(aux_x0[smallest_ind]).cuda(), requires_grad=False)


    net = Net_v02(init_LOC_weights = init_LOC_weights_np).cuda()

    optimizer = th.optim.RMSprop([net.LOC_weights], lr=0.1)
    loss_func = th.nn.MSELoss()

    for epoch in range(800):

        pred = net(sample, x0, steps = 20, N = N)

        loss = loss_func(pred, alpha)

        optimizer.zero_grad()           # clear gradients for this training step

        loss.backward(retain_graph=True)                 # backpropagation, compute gradients
        # loss.backward()
        # clip_grad_norm([net.x1, net.x2, net.x3], max_norm=0.1, norm_type=2)
        optimizer.step()

        # auxiliary computation to avoid large number of steps in CG
        aux_x0 = []
        aux_err = []
        print(net.LOC_weights.data)
        for _ in range(3):
            x0, err = validation_v3(sample, x0, alpha, param = net.LOC_weights.cpu().data.numpy(), valid_steps = 100)
            aux_x0.append(x0.cpu().data.numpy().astype(np.float64))
            aux_err.append(err)

        smallest_ind = np.argmin(aux_err)
        x0 = Variable(th.from_numpy(aux_x0[smallest_ind]).cuda(), requires_grad=False)

        if epoch % 100 ==0:
            print('epoch at %02d !!!' % epoch)

            print('grad')
            print(sum(net.LOC_weights.grad.cpu().data.numpy()))
            print('params')
            print(sum(net.LOC_weights.cpu().data.numpy()))

            validation_v3(sample, x0, alpha, param = net.LOC_weights.cpu().data.numpy(), valid_steps = 100)
            print(loss.abs())
            tmp_file_name = tmp_dir_name + ('/file_%02d_v21c_epoch_%04d' % ((file_id+1), epoch))
            np.save(tmp_file_name, aux_x0[smallest_ind])

            weights = net.LOC_weights.cpu().data.numpy()
            tmp_file_name = tmp_dir_name + ('/file_%02d_v21c_LOCweights_%04d' % ((file_id+1), epoch))
            np.save(tmp_file_name, weights)
