"""
Image => CNN => sparse weights (Lcm/Lmat/Luu) => better image matting results
Opimise weights for alpha computation in image matting in Sparse fashion
Author: Shu Liu
Email: liush@ethz.ch
devel log: v22f.py
Description: Description: use CNN to compute Lcm, Lmat, Luu(sparse weigths) from flow *.mat files and optimize CNN
    by computing (use sparce cg) matting and comparing with the ground truth matting
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

gt_prefix = 'Training/alpha/GT'
trimap_prefix = 'Training/trimap/GT'
image_prefix = 'Training/images/GT'


def validation_v3(sample, x0, alpha, param = None, valid_steps = 100, thresh = 1e-10):
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
    b = sp.spmv(sp.spadd(KU, known), kToU)

    res, _, _ = optim.sparse_cg_ib2(Asys, b, x0, steps = valid_steps, thresh = thresh)

    alpha = np.reshape(sample['matte'], [-1], 'F')
    alpha = Variable(th.from_numpy(alpha.astype(np.float64)).cuda(), requires_grad=False)

    tmp_err = np.mean((res.cpu().data.numpy()-alpha.cpu().data.numpy())**2)
    print('validation loss:', tmp_err)

    np.save('debug_DataLoader', res.cpu().data.numpy())
    return res, tmp_err


def data_from_idx(idx):

    file_id = ('%02d' % idx)

    gt_path = gt_prefix + file_id + '.png'
    trimap_path = trimap_prefix + file_id + '.png'
    img_path = image_prefix + file_id + '.png'

    gt = np.asarray(imageio.imread(gt_path)[:,:,0]) / 255.0
    trimap = np.expand_dims(np.asarray(imageio.imread(trimap_path)), axis = 2)
    img = np.asarray(imageio.imread(img_path))

    img_trimap = np.concatenate((img, trimap), 2)
    img_trimap = np.expand_dims(img_trimap, axis = 0)
    img_trimap = np.moveaxis(img_trimap, [3], [1])

    weight_mask = np.asarray(imageio.imread(trimap_path))
    weight_mask = np.asarray(list(weight_mask)).astype(np.float64)
    weight_mask = (weight_mask == 128).astype(np.float64)

    return img_trimap, weight_mask


class Net_v02(th.nn.Module):
    def __init__(self):
        super(Net_v02, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Sigmoid(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )


    def forward(self, img_trimap, mask, sample, x0, steps, N):

        x = Variable(th.from_numpy(img_trimap.astype(np.float64)).cuda(), requires_grad=False)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        print('x shape', x.shape)

        mask_th = Variable(th.from_numpy(mask.astype(np.float64)).cuda(), requires_grad=False)

        print('mask_th shape', mask_th.shape)

        self.masked_weight = th.mul(x4, mask_th)

        self.masked_weight = self.masked_weight.view(-1)

        print('masked_weight shape', self.masked_weight.shape)

        lmbda = Variable(th.from_numpy(np.array([100.0], dtype=np.float64)).cuda(),
                         requires_grad=False)

        Lmat_sp =  matting_sys._matting_laplacian(N = N, sample = sample, LOC_weights = self.masked_weight)
        print('Lmat_sp', Lmat_sp.size)

        kToUconf = sample['kToUconf'].view(-1)
        known = sample['known'].view(-1)
        kToU = sample['kToU'].view(-1)
        linear_idx = Variable(th.from_numpy(np.arange(N, dtype=np.int32)).cuda())
        linear_csr_row_idx = Variable(th.from_numpy(np.arange(N+1, dtype=np.int32)).cuda())

        self.KU_weights = Variable(th.from_numpy(np.ones(N).astype(np.float64)*0.05).cuda(), requires_grad=False)
        KU = sp.Sparse(linear_csr_row_idx, linear_idx, self.KU_weights.mul(kToUconf), th.Size((N,N)))
        known = sp.Sparse(linear_csr_row_idx, linear_idx, lmbda.mul(known), th.Size((N,N)))

        Asys = sp.spadd(Lmat_sp, sp.spadd(KU, known))
        b = sp.spmv(sp.spadd(KU, known), kToU)

        x0, _, _ = optim.sparse_cg_ib2(Asys, b, x0, steps = steps, thresh = 1e-16)

        return x0


net = Net_v02().cuda()

for file_id in range(3,4):

    print('File %02d' % (file_id + 1) )

    img_trimap, weight_mask = data_from_idx(file_id+1)

    input_ima_trimap = Variable(th.from_numpy(img_trimap.astype(np.float64)), requires_grad=False)

    sample = dataloader.__getitem__(file_id)
    alpha = np.reshape(sample['matte'], [-1], 'F')
    alpha = Variable(th.from_numpy(alpha.astype(np.float64)).cuda(), requires_grad=False)

    N = sample['image'].shape[0] * sample['image'].shape[1]

    x0 = Variable(th.from_numpy(np.zeros(N).astype(np.float64)).cuda(), requires_grad=False)

    init_LOC_weights_np  = np.ones(N).astype(np.float64)

    aux_x0 = []
    aux_err = []
    for _ in range(5):
        pred, err = validation_v3(sample, x0, alpha, param = init_LOC_weights_np, valid_steps = 100, thresh = 1e-16)
        x0 = Variable(th.from_numpy(pred.cpu().data.numpy().astype(np.float64)).cuda(), requires_grad=False)
        aux_x0.append(x0.cpu().data.numpy().astype(np.float64))
        aux_err.append(err)

    smallest_ind = np.argmin(aux_err)
    init_x0 = Variable(th.from_numpy(aux_x0[smallest_ind]).cuda(), requires_grad=False)
    init_err = aux_err[smallest_ind]
    print('init_err', init_err)


    for parameter in net.parameters():
        print(parameter)

    optimizer = th.optim.RMSprop(net.parameters(), lr=0.1)
    loss_func = th.nn.MSELoss()

    img_trimap, weight_mask = data_from_idx(file_id+1)

    nn_err = 1000.0
    no_init_flag = False
    for epoch in range(901):

        if not no_init_flag:
            x0 = Variable(th.from_numpy(np.zeros(N).astype(np.float64)).cuda(), requires_grad=False)
            aux_x0 = []
            aux_err = []
            for _ in range(2):
                pred, err = validation_v3(sample, x0, alpha, param = init_LOC_weights_np, valid_steps = 100, thresh = 1e-16)
                x0 = Variable(th.from_numpy(pred.cpu().data.numpy().astype(np.float64)).cuda(), requires_grad=False)
                aux_x0.append(x0.cpu().data.numpy().astype(np.float64))
                aux_err.append(err)
            smallest_ind = np.argmin(aux_err)
            init_err = aux_err[smallest_ind]
            if init_err > nn_err:
                print('yep, init_err: %f nn_err: %f' % (init_err, nn_err))
                x0 = nn_x0
                no_init_flag = True

        pred = net(img_trimap, weight_mask, sample, x0, steps = 50, N = N)

        loss = loss_func(pred, alpha)

        optimizer.zero_grad()           # clear gradients for this training step
        # print('before loss_backward and optimizer_step %f, %f' % (net.x1, net.x2))

        loss.backward(retain_graph=False)                 # backpropagation, compute gradients
        # loss.backward()
        # clip_grad_norm([net.x1, net.x2, net.x3], max_norm=0.1, norm_type=2)
        optimizer.step()

        # avoid endless graph, which leads to memory explosion
        # x0 = Variable(th.from_numpy(pred.cpu().data.numpy().astype(np.float64)).cuda(), requires_grad=False)

        # auxiliary computation to avoid large number of steps in CG
        aux_x0 = []
        aux_err = []
        # print(net.LOC_weights.data)
        for j in range(3):
            print('round', j)
            x0, err = validation_v3(sample, x0, alpha, param = net.masked_weight.cpu().data.numpy(), valid_steps = 100)
            aux_x0.append(x0.cpu().data.numpy().astype(np.float64))
            aux_err.append(err)

        smallest_ind = np.argmin(aux_err)
        nn_err = aux_err[smallest_ind]
        nn_x0 = Variable(th.from_numpy(aux_x0[smallest_ind]).cuda(), requires_grad=False)
        print('MSE:', aux_err[smallest_ind])

        if epoch % 5 == 0:
            print('epoch at %02d !!!' % epoch)
            print('params')
            print(sum(net.masked_weight.cpu().data.numpy().reshape([-1])))

        if epoch % 100 ==0:
            print('epoch at %02d !!!' % epoch)
            print('params')
            print(sum(net.masked_weight.cpu().data.numpy().reshape([-1])))
            validation_v3(sample, x0, alpha, param = net.masked_weight.cpu().data.numpy(), valid_steps = 100)
            print(loss.abs())
