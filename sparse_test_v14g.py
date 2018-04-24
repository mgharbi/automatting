"""
Opimise weights for alpha computation in image matting in Sparse fashion
Author: Shu Liu
Email: liush@student.ethz.ch
devel log: v14g.py
Description: optimise two weights for Lcm, Lmat and a diag. matrix for Luu; use matrix inverse of pytorch. It works.
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

raw_data = spio.loadmat('smallSys/smallSysData.mat', squeeze_me=True)
Atri_raw = raw_data['Atri']
Lcm_raw = raw_data['Lcm']
Lmat_raw = raw_data['Lmat']
Lmat_raw = 0.5*(Lmat_raw + Lmat_raw.transpose())
Luu_raw = raw_data['Luu']
b_raw = raw_data['b']
b_raw = b_raw.reshape([1600])
alpha = imageio.imread('smallSys/gnd.png')
alpha = np.asarray(alpha).transpose()
alpha_raw = alpha.reshape([1600]) / 255.0 # normalization to range [0, 1]

diag = np.diag([0.1 for _ in range(1600)])
A = Atri_raw + Lcm_raw + Lmat_raw +  np.matmul(np.matmul(diag, Luu_raw), diag)
# A = Atri + Lcm + Lmat + Luu
alpha2 = np.dot(np.linalg.inv(A), b_raw)
print(np.mean((alpha2 - alpha_raw)**2))

# option to load TF optimized parameters
elem_wise = spio.loadmat('dense_elementwise_weight.mat', squeeze_me=True)
param1 = elem_wise['weight_lcm']
param2 = elem_wise['weight_lmat']
param3 = elem_wise['weight_luu']

param1 = 1.0
param2 = 1.0
param3 = np.diag([0.1 for _ in range(1600)])

Atri = Variable(th.from_numpy(Atri_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
Lcm = Variable(th.from_numpy(Lcm_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
Lcm_init= Variable(th.from_numpy((param1 * Lcm_raw).astype(np.float32).reshape([1,1,1600,1600])).cuda())
Lmat = Variable(th.from_numpy(Lmat_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
Lmat_init = Variable(th.from_numpy((param2 * Lmat_raw).astype(np.float32).reshape([1,1,1600,1600])).cuda())
Luu = Variable(th.from_numpy(Luu_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
Luu_init = Variable(th.from_numpy((np.matmul(np.matmul(param3, Luu_raw), param3)).astype(np.float32).reshape([1,1,1600,1600])).cuda())
b = Variable(th.from_numpy(b_raw.astype(np.float32)))
alpha = Variable(th.from_numpy(alpha_raw.astype(np.float32)).cuda())

# convert torch Variable in sparse_cg
Atri_sp = ssp.csr_matrix(Atri.cpu().data.numpy()[0,0]).tocoo()
Atri_sp = sp.from_coo(Variable(th.from_numpy(Atri_sp.row).cuda(), requires_grad=False), Variable(th.from_numpy(Atri_sp.col).cuda(), requires_grad=False), Variable(th.from_numpy(Atri_sp.data.astype(np.float32)).cuda(), requires_grad=False), th.Size((Atri_sp.shape[0], Atri_sp.shape[1])))

Lcm_sp = ssp.csr_matrix(Lcm.cpu().data.numpy()[0,0]).tocoo()
Lcm_sp = sp.from_coo(Variable(th.from_numpy(Lcm_sp.row).cuda()), Variable(th.from_numpy(Lcm_sp.col).cuda()), Variable(th.from_numpy(Lcm_sp.data.astype(np.float32)).cuda()), th.Size((Lcm_sp.shape[0], Lcm_sp.shape[1])))

Lcm_init_sp = ssp.csr_matrix(Lcm_init.cpu().data.numpy()[0,0]).tocoo()
Lcm_init_sp = sp.from_coo(Variable(th.from_numpy(Lcm_init_sp.row).cuda()), Variable(th.from_numpy(Lcm_init_sp.col).cuda()), Variable(th.from_numpy(Lcm_init_sp.data.astype(np.float32)).cuda()), th.Size((Lcm_init_sp.shape[0], Lcm_init_sp.shape[1])))

Lmat_sp = ssp.csr_matrix(Lmat.cpu().data.numpy()[0,0]).tocoo()
Lmat_sp = sp.from_coo(Variable(th.from_numpy(Lmat_sp.row).cuda()), Variable(th.from_numpy(Lmat_sp.col).cuda()), Variable(th.from_numpy(Lmat_sp.data.astype(np.float32)).cuda()), th.Size((Lmat_sp.shape[0], Lmat_sp.shape[1])))

Lmat_init_sp = ssp.csr_matrix(Lmat_init.cpu().data.numpy()[0,0]).tocoo()
Lmat_init_sp = sp.from_coo(Variable(th.from_numpy(Lmat_init_sp.row).cuda()), Variable(th.from_numpy(Lmat_init_sp.col).cuda()), Variable(th.from_numpy(Lmat_init_sp.data.astype(np.float32)).cuda()), th.Size((Lmat_init_sp.shape[0], Lmat_init_sp.shape[1])))

Luu_sp = ssp.csr_matrix(Luu.cpu().data.numpy()[0,0]).tocoo()
Luu_sp = sp.from_coo(Variable(th.from_numpy(Luu_sp.row).cuda()), Variable(th.from_numpy(Luu_sp.col).cuda()), Variable(th.from_numpy(Luu_sp.data.astype(np.float32)).cuda()), th.Size((Luu_sp.shape[0], Luu_sp.shape[1])))

Luu_init_sp = ssp.csr_matrix(Luu_init.cpu().data.numpy()[0,0]).tocoo()
Luu_init_sp = sp.from_coo(Variable(th.from_numpy(Luu_init_sp.row).cuda()), Variable(th.from_numpy(Luu_init_sp.col).cuda()), Variable(th.from_numpy(Luu_init_sp.data.astype(np.float32)).cuda()), th.Size((Luu_init_sp.shape[0], Luu_init_sp.shape[1])))

b_sp = Variable(th.from_numpy(b.data.numpy().astype(np.float32)).cuda(), requires_grad=False)
x0 = Variable(th.from_numpy(np.zeros([b.shape[0]]).astype(np.float32)).cuda(), requires_grad=False) # will be updated
x_zero = Variable(th.from_numpy(np.zeros([b.shape[0]]).astype(np.float32)).cuda(), requires_grad=False) # will never be updated

print('x0 init', sum(x0.cpu().data.numpy()))

# conjugate gradient solver
def cg_solver(Atri, Lcm, Lmat, Luu, b, x0, steps = 1, thresh = 1e-7):
    A_sys = sp.spadd(Atri, Lcm)
    A_sys = sp.spadd(A_sys, Lmat)
    A_sys = sp.spadd(A_sys, Luu)
    tmp = optim.sparse_cg(A_sys, b, x0, steps = steps, thresh = thresh)
    return tmp

for init_run in range(5):
    x0, _, _ = cg_solver(Atri_sp, Lcm_init_sp, Lmat_init_sp, Luu_init_sp, b_sp, x0, steps = 100000, thresh =1e-16)
    if init_run % 2 == 0:
        print('init loss:', np.mean((x0.cpu().data.numpy()-alpha.cpu().data.numpy())**2))

def validation(Atri, Lcm, Lmat, Luu, b, x0, alpha, param1, param2, param3, valid_steps = 2000):
    Lcm.mul_(param1)
    Lmat.mul_(param2)
    # Luu.mul_(param3)

    tmp_x3 = np.diag([0.1 for _ in range(1600)])
    tmp_x3 = ssp.csr_matrix(tmp_x3).tocoo()
    param3_mat = sp.from_coo(Variable(th.from_numpy(tmp_x3.row).cuda(), requires_grad=False), Variable(th.from_numpy(tmp_x3.col).cuda(), requires_grad=False), param3, th.Size((tmp_x3.shape[0], tmp_x3.shape[1])))
    Luu = sp.spmm(param3_mat, sp.spmm(Luu, param3_mat))

    tmp_res, _, _ = cg_solver(Atri, Lcm, Lmat, Luu, b, x0, steps = valid_steps, thresh = 1e-12)
    print('validation loss:', np.mean((tmp_res.cpu().data.numpy()-alpha.cpu().data.numpy())**2))


def validation_v2(Atri, Lcm, Lmat, Luu, b, x0, alpha, param1, param2, param3, valid_steps = 5000):
    # Asys = Atri + param1 * Lcm + param2 * Lmat + np.matmul(np.matmul(param3, Luu), param3)
    param3 = np.diag(param3)
    Asys = Atri + param1 * Lcm + param2 * Lmat + np.matmul(np.matmul(param3, Luu), param3)

    Asys = Variable(th.from_numpy(Asys.astype(np.float32).reshape([1,1,1600,1600])).cuda())

    Asys_sp = ssp.csr_matrix(Asys.cpu().data.numpy()[0,0]).tocoo()
    Asys_sp = sp.from_coo(Variable(th.from_numpy(Asys_sp.row).cuda(), requires_grad=False), Variable(th.from_numpy(Asys_sp.col).cuda(), requires_grad=False), Variable(th.from_numpy(Asys_sp.data.astype(np.float32)).cuda(), requires_grad=False), th.Size((Asys_sp.shape[0], Asys_sp.shape[1])))

    b =  Variable(th.from_numpy(b.astype(np.float32)))
    b_sp = Variable(th.from_numpy(b.data.numpy().astype(np.float32)).cuda(), requires_grad=False)

    x0, _, _ = optim.sparse_cg(Asys_sp, b_sp, x0, steps = valid_steps, thresh = 1e-16)

    print('validation loss:', np.mean((x0.cpu().data.numpy()-alpha.cpu().data.numpy())**2))


class Net_v02(th.nn.Module):
    def __init__(self):
        super(Net_v02, self).__init__()

        self.x1 = Variable(th.Tensor([param1]).cuda(), requires_grad=True)   # weight for Lcm

        self.x2 = Variable(th.Tensor([param2]).cuda(), requires_grad=True)   # weight for Lmat

        self.x3 = Variable(th.Tensor([0.1 for _ in range(1600)]).cuda(), requires_grad=True)   # weight for Luu


    def forward(self, Atri, Lcm, Lmat, Luu, b, x0, steps):

        Atri = Variable(th.from_numpy(Atri.astype(np.float32)).cuda())
        Lcm = th.mul(Variable(th.from_numpy(Lcm.astype(np.float32)).cuda()), self.x1)
        Lmat = th.mul(Variable(th.from_numpy(Lmat.astype(np.float32)).cuda()), self.x2)
        diag = th.diag(self.x3)
        Luu = th.matmul(th.matmul(diag, Variable(th.from_numpy(Luu.astype(np.float32)).cuda())), diag)

        b = Variable(th.from_numpy(b.astype(np.float32)).cuda())

        A = Atri + Lcm + Lmat + Luu
        res = th.matmul(th.inverse(A), b)

        return res

net = Net_v02().cuda()

optimizer = th.optim.SGD([net.x1, net.x2, net.x3], lr=1.0)
loss_func = th.nn.MSELoss()

for epoch in range(100000):

    Atri = Variable(th.from_numpy(Atri_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
    Lcm = Variable(th.from_numpy(Lcm_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
    Lmat = Variable(th.from_numpy(Lmat_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
    Luu = Variable(th.from_numpy(Luu_raw.astype(np.float32).reshape([1,1,1600,1600])).cuda())
    b = Variable(th.from_numpy(b_raw.astype(np.float32)))

    pred = net(Atri_raw, Lcm_raw, Lmat_raw, Luu_raw, b_raw, x0, steps = 200)

    x0 = Variable(th.from_numpy(pred.cpu().data.numpy().astype(np.float32)).cuda(), requires_grad=False) # avoid endless graph, which leads to memory explosion

    loss = loss_func(pred, alpha)

    optimizer.zero_grad()           # clear gradients for this training step

    loss.backward(retain_graph=True)                 # backpropagation, compute gradients

    optimizer.step()
    print('grad')
    print(net.x1.grad.data[0], net.x2.grad.data[0], sum(net.x3.grad.cpu().data.numpy()))

    if (net.x1.grad.data[0] == 0 and net.x2.grad.data[0] == 0 and sum(net.x3.grad.cpu().data.numpy()) == 0):
        print('!!!optimal parameter found!!!')
        print('optimal params')
        print(net.x1.data[0], net.x2.data[0], sum(net.x3.data))
        print('final loss')
        validation_v2(Atri_raw, Lcm_raw, Lmat_raw, Luu_raw, b_raw, x0, alpha, net.x1.data[0], net.x2.data[0], net.x3.data, valid_steps = 200)
        print(loss.abs())
        break

    if epoch % 10 ==0:
        print('epoch at %02d !!!' % epoch)

        print('grad')
        print(net.x1.grad.data[0], net.x2.grad.data[0], sum(net.x3.grad.cpu().data.numpy()))
        print('params')
        print(net.x1.data[0], net.x2.data[0], net.x3.data[0])

        validation_v2(Atri_raw, Lcm_raw, Lmat_raw, Luu_raw, b_raw, x_zero, alpha, net.x1.data[0], net.x2.data[0], net.x3.data, valid_steps = 10000)

        # naive numpy matrix inverse solution
        diag = np.diag(net.x3.data)
        A = Atri_raw + Lcm_raw * net.x1.data[0] + Lmat_raw * net.x2.data[0] +  np.matmul(np.matmul(diag, Luu_raw), diag)
        alpha2 = np.dot(np.linalg.inv(A), b_raw)
        print('naive numpy inverse',np.mean((alpha2 - alpha_raw)**2))
        # end of naive numpy matrix inverse solution

        # save intermediate results
        np.save('param1', net.x1.data[0])
        np.save('param2', net.x2.data[0])
        np.save('param3', net.x3.data)
        np.save('pytorch_res', pred.data)

        print(loss.abs())
