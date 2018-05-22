"""
Opimise weights for alpha computation in image matting in Sparse fashion
Author: Shu Liu
Email: liush@student.ethz.ch
devel log: v13k.py
Description: optimise two weights for Lcm, Lmat and one diag. matrix for Luu; use updated conjugate gradient methods(sparse_cg_ib2), where
    we take intermediate(i) best(b) results for backpropagation, restrict return idx of result to be greater equal to 5;
    optimisation steps in BP is restricted to 20;
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

log = logging.getLogger(__name__)

th.set_default_tensor_type('torch.DoubleTensor')

raw_large_sys = spio.loadmat('largesys.mat', squeeze_me=True)
Atri_large = raw_large_sys['Atri'].astype(np.float64)
Lcm_large = raw_large_sys['Lcm'].astype(np.float64)
Lmat_large = raw_large_sys['Lmat'].astype(np.float64)
Luu_large = raw_large_sys['Luu'].astype(np.float64)
alpha_large = raw_large_sys['alphaHat'].astype(np.float64)
conf_large = raw_large_sys['conf'].astype(np.float64)

mat_size = Atri_large.shape[0]

alpha = spio.loadmat('gnd.mat', squeeze_me = True)
alpha = alpha['ground']

Atri_sp = sp.Sparse(Variable(th.from_numpy(Atri_large.indptr).cuda()), Variable(th.from_numpy(Atri_large.indices).cuda()), Variable(th.from_numpy(Atri_large.data).cuda()), th.Size((Atri_large.shape[0], Atri_large.shape[1])))
Lcm_sp = sp.Sparse(Variable(th.from_numpy(Lcm_large.indptr).cuda()), Variable(th.from_numpy(Lcm_large.indices).cuda()), Variable(th.from_numpy(Lcm_large.data).cuda()), th.Size((Lcm_large.shape[0], Lcm_large.shape[1])))
Lmat_sp = sp.Sparse(Variable(th.from_numpy(Lmat_large.indptr).cuda()), Variable(th.from_numpy(Lmat_large.indices).cuda()), Variable(th.from_numpy(Lmat_large.data).cuda()), th.Size((Lmat_large.shape[0], Lmat_large.shape[1])))
Luu_sp = sp.Sparse(Variable(th.from_numpy(Luu_large.indptr).cuda()), Variable(th.from_numpy(Luu_large.indices).cuda()), Variable(th.from_numpy(Luu_large.data).cuda()), th.Size((Luu_large.shape[0], Luu_large.shape[1])))

Conf_sp = ssp.csr_matrix(ssp.spdiags(conf_large, 0, len(conf_large), len(conf_large))).tocoo()
Conf_sp = sp.from_coo(Variable(th.from_numpy(Conf_sp.row).cuda()), Variable(th.from_numpy(Conf_sp.col).cuda()), Variable(th.from_numpy(Conf_sp.data.astype(np.float64)).cuda()), th.Size((Conf_sp.shape[0], Conf_sp.shape[1])))

b = Variable(th.from_numpy(conf_large.astype(np.float64)))
b_sp = Variable(th.from_numpy(b.data.numpy().astype(np.float64)).cuda(), requires_grad=False)

alpha_hat = Variable(th.from_numpy(alpha_large.astype(np.float64)).cuda(), requires_grad=False)
alpha = Variable(th.from_numpy(alpha.astype(np.float64)).cuda(), requires_grad=False)

x0 = Variable(th.from_numpy(np.zeros([b.shape[0]]).astype(np.float64)).cuda(), requires_grad=False)
x_zero = Variable(th.from_numpy(np.zeros([b.shape[0]]).astype(np.float64)).cuda(), requires_grad=False)


# conjugate gradient solver
def cg_solver(Atri, Lcm, Lmat, Luu, Conf, alpha_hat, b, x0, steps = 1, thresh = 1e-8):
    A_star = sp.spadd(Lcm, Lmat)
    A_star = sp.spadd(A_star, Luu)
    A_sys = sp.spadd(A_star, Atri)
    A_sys = sp.spadd(A_sys, Conf)
    tmp = sp.spadd(Atri, Conf)
    b = sp.spmv(tmp, alpha_hat)
    tmp = optim.sparse_cg_ib2(A_sys, b, x0, steps = steps, thresh = thresh)
    return tmp[0]

for init_run in range(3):
    x0 = cg_solver(Atri_sp, Lcm_sp, Lmat_sp, Luu_sp, Conf_sp, alpha_hat, b_sp, x0, steps = 400)
    print('init loss at run : %02d' % init_run , np.mean((x0.cpu().data.numpy()-alpha.cpu().data.numpy())**2))

res_one_run = cg_solver(Atri_sp, Lcm_sp, Lmat_sp, Luu_sp, Conf_sp, alpha_hat, b_sp, x_zero, steps = 1600, thresh = 1e-16)
print('init loss at one-run :' , np.mean((res_one_run.cpu().data.numpy()-alpha.cpu().data.numpy())**2))

def validation(Atri, Lcm, Lmat, Luu, Conf, alpha_hat, b, x0, alpha, param1, param2, param3, valid_steps = 100):
    Lcm.mul_(param1)
    Lmat.mul_(param2)

    mat_idx = np.array([i for i in range(mat_size)]).astype(np.int32)
    mat_param3 = sp.from_coo(Variable(th.from_numpy(mat_idx).cuda(), requires_grad=False), Variable(th.from_numpy(mat_idx).cuda(), requires_grad=False), param3, th.Size((mat_size, mat_size)))

    Luu = sp.spmm(mat_param3, sp.spmm(Luu, mat_param3))
    res = cg_solver(Atri, Lcm, Lmat, Luu, Conf, alpha_hat, b, x0, steps = valid_steps)
    tmp_err = np.mean((res.cpu().data.numpy()-alpha.cpu().data.numpy())**2)
    print('validation loss:', tmp_err)
    return res, tmp_err

class Net_v02(th.nn.Module):
    def __init__(self):
        super(Net_v02, self).__init__()

        self.x1 = Variable(th.Tensor([1.0]).cuda(), requires_grad=True)   # weight for Lcm

        self.x2 = Variable(th.Tensor([1.0]).cuda(), requires_grad=True)   # weight for Lmat

        mat_idx = np.array([i for i in range(mat_size)]).astype(np.int32)

        self.x3 = Variable(th.from_numpy(np.array([0.1 for _ in range(mat_size)]).astype(np.float64)).cuda(), requires_grad=True)
        self.mat_x3 = sp.from_coo(Variable(th.from_numpy(mat_idx).cuda(), requires_grad=False), Variable(th.from_numpy(mat_idx).cuda(), requires_grad=False), self.x3, th.Size((mat_size, mat_size)))


    def forward(self, Atri, Lcm, Lmat, Luu, Conf, ahlph_hat, b, x0, steps):

        Lcm.mul_(self.x1)
        Lmat.mul_(self.x2)

        Luu = sp.spmm(self.mat_x3, sp.spmm(Luu, self.mat_x3))

        A_star = sp.spadd(Lcm, Lmat)
        A_star = sp.spadd(A_star, Luu)
        A_sys = sp.spadd(A_star, Atri)
        A_sys = sp.spadd(A_sys, Conf)
        tmp = sp.spadd(Atri, Conf)
        b = sp.spmv(tmp, alpha_hat)

        res, _, _ = optim.sparse_cg_ib2(A_sys, b, x0, steps = steps, thresh = 1e-9)

        return res

net = Net_v02().cuda()

# optimizer = th.optim.Adagrad([net.x1, net.x2, net.x3], lr=0.0001)
optimizer = th.optim.RMSprop([net.x1, net.x2, net.x3], lr=0.0001)
loss_func = th.nn.MSELoss()


for epoch in range(1000):

    Atri_sp = sp.Sparse(Variable(th.from_numpy(Atri_large.indptr).cuda()), Variable(th.from_numpy(Atri_large.indices).cuda()), Variable(th.from_numpy(Atri_large.data).cuda()), th.Size((Atri_large.shape[0], Atri_large.shape[1])))
    Lcm_sp = sp.Sparse(Variable(th.from_numpy(Lcm_large.indptr).cuda()), Variable(th.from_numpy(Lcm_large.indices).cuda()), Variable(th.from_numpy(Lcm_large.data).cuda()), th.Size((Lcm_large.shape[0], Lcm_large.shape[1])))
    Lmat_sp = sp.Sparse(Variable(th.from_numpy(Lmat_large.indptr).cuda()), Variable(th.from_numpy(Lmat_large.indices).cuda()), Variable(th.from_numpy(Lmat_large.data).cuda()), th.Size((Lmat_large.shape[0], Lmat_large.shape[1])))
    Luu_sp = sp.Sparse(Variable(th.from_numpy(Luu_large.indptr).cuda()), Variable(th.from_numpy(Luu_large.indices).cuda()), Variable(th.from_numpy(Luu_large.data).cuda()), th.Size((Luu_large.shape[0], Luu_large.shape[1])))

    Conf_sp = ssp.csr_matrix(ssp.spdiags(conf_large, 0, len(conf_large), len(conf_large))).tocoo()
    Conf_sp = sp.from_coo(Variable(th.from_numpy(Conf_sp.row).cuda()), Variable(th.from_numpy(Conf_sp.col).cuda()), Variable(th.from_numpy(Conf_sp.data.astype(np.float64)).cuda()), th.Size((Conf_sp.shape[0], Conf_sp.shape[1])))

    b = Variable(th.from_numpy(conf_large.astype(np.float64)))
    b_sp = Variable(th.from_numpy(b.data.numpy().astype(np.float64)).cuda(), requires_grad=False)
    alpha_hat = Variable(th.from_numpy(alpha_large.astype(np.float64)).cuda(), requires_grad=False)

    pred = net(Atri_sp, Lcm_sp, Lmat_sp, Luu_sp, Conf_sp, alpha_hat, b_sp, x0, steps = 20)


    loss = loss_func(pred, alpha)
    optimizer.zero_grad()           # clear gradients for this training step

    loss.backward(retain_graph=True)

    optimizer.step()

    # auxiliary computation to avoid large number of steps in CG
    aux_x0 = []
    aux_res = []
    for _ in range(3):
        x0, res = validation(Atri_sp, Lcm_sp, Lmat_sp, Luu_sp, Conf_sp, alpha_hat, b_sp, x0, alpha, net.x1, net.x2, net.x3, valid_steps = 100)
        aux_x0.append(x0.cpu().data.numpy().astype(np.float64))
        aux_res.append(res)

    smallest_ind = np.argmin(aux_res)
    x0 = Variable(th.from_numpy(aux_x0[smallest_ind]).cuda(), requires_grad=False)

    print('grad')
    print(net.x1.grad.data[0], net.x2.grad.data[0], sum(net.x3.grad.cpu().data.numpy()))

    if epoch % 10 ==0:
        print('epoch at %02d !!!' % epoch)
        tmp_file_name = ('v13k_res/epoch_v2_%04d' % epoch)
        np.save(tmp_file_name, aux_x0[smallest_ind])
    if epoch % 30 == 0:
        weights = list(net.x1.cpu().data.numpy()) + list(net.x2.cpu().data.numpy()) + list(net.x3.cpu().data.numpy())
        tmp_file_name = ('v13k_res/epoch_weights_%04d' % epoch)
        np.save(tmp_file_name, weights)
