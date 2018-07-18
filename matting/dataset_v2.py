"""
Data Loader
Author: Shu Liu
Email: liush@ethz.ch
Description: based on dataset.py; load flows(*.mat) that are precomputed in matlab;
    load data in a more consistent way as matlab (easy to compare between matlab and python)
"""
import logging
import os
import re
import time

import numpy as np
import scipy.io
import scipy.sparse as sp
import skimage.io
import torch as th
from torch.utils.data import Dataset

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

log = logging.getLogger(__name__)

th.set_default_tensor_type('torch.DoubleTensor')

class MattingDataset(Dataset):
  """"""

  def __init__(self, root_dir, transform=None):
    super(MattingDataset, self).__init__()
    self.transform = transform

    self.root_dir = root_dir
    self.ifm_data_dir = os.path.join(root_dir, 'IFMData')
    # self.ifm_data_dir = os.path.join(root_dir, 'IFMData1_overkill')
    self.matte_dir = os.path.join(root_dir, 'alpha')
    self.images_dir = os.path.join(root_dir, 'images')
    self.trimap_dir = os.path.join(root_dir, 'trimap')
    # self.vanilla_dir = os.path.join(root_dir, 'vanilla')

    files = os.listdir(self.images_dir)
    data_regex = re.compile(r".*.(png|jpg|jpeg)$")
    files = sorted([f for f in files if data_regex.match(f)])

    fid = open("missing_files.txt", 'w')

    start = time.time()

    self.files = []
    for f in files:
      if not os.path.exists(self.ifm_path(f)):
        fid.write(self.ifm_path(f))
        fid.write("\n")
        continue
      if not os.path.exists(self.matte_path(f)):
        fid.write(self.matte_path(f))
        fid.write("\n")
        continue
      if not os.path.exists(self.trimap_path(f)):
        fid.write(self.trimap_path(f))
        fid.write("\n")
        continue
      self.files.append(f)
    fid.close()

    duration = time.time() - start

    log.debug("Parsed dataset {} with {} samples in {:.2f}s".format(
      root_dir, len(self), duration))


  def image_path(self, f):
    return os.path.join(self.images_dir, f)

  def basename(self, f):
    fname = os.path.splitext(f)[0]
    basename = fname
    # basename = "_".join(fname.split("_")[:-1])
    return basename

  def ifm_path(self, f):
    return os.path.join(self.ifm_data_dir, os.path.splitext(f)[0]+".mat")

  def matte_path(self, f):
    return os.path.join(self.matte_dir, self.basename(f)+".png")


  def trimap_path(self, f):
    return os.path.join(self.trimap_dir, self.basename(f)+".png")

  def result_path(self, f):
    return os.path.join('', self.basename(f)+"/")


  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    start = time.time()
    fname = self.files[idx]

    matte = skimage.io.imread(self.matte_path(fname)).astype(np.float64)[:, :, 0:1]/255.0
    image = skimage.io.imread(self.image_path(fname)).astype(np.float64)/255.0
    trimap = skimage.io.imread(self.trimap_path(fname)).astype(np.float64)/255.0

    raw_data = scipy.io.loadmat(self.ifm_path(fname))
    N = image.shape[0] * image.shape[1]

    sample = {}
    sample['image'] = Variable(th.from_numpy(image).cuda())
    N = sample['image'].shape[0] * sample['image'].shape[1]
    sample['CM_inInd'] = Variable(th.from_numpy(raw_data['CM_inInd'].astype(np.int32)-1).cuda(), requires_grad=False) # index start from 1 in matlab, from 0 in python
    sample['CM_neighInd'] = Variable(th.from_numpy(raw_data['CM_neighInd'].astype(np.int32)-1).cuda(), requires_grad=False) # index start from 1 in matlab, from 0 in python
    sample['CM_flows'] = Variable(th.from_numpy(raw_data['CM_flows'].astype(np.float64)).cuda(), requires_grad=False)

    sample['kToUconf'] = Variable(th.from_numpy(np.reshape(raw_data['kToUconf'].astype(np.float64), [N], 'F')).cuda(), requires_grad=False) #‘F’ means to read / write the elements using Fortran-like index order; keep the same as matlab
    sample['kToU'] = Variable(th.from_numpy(np.reshape(raw_data['kToU'].astype(np.float64), [N], 'F')).cuda(), requires_grad=False) #‘F’ means to read / write the elements using Fortran-like index order; keep the same as matlab
    sample['known'] = Variable(th.from_numpy(np.reshape(raw_data['known'].astype(np.float64), [N], 'F')).cuda(), requires_grad=False)   #‘F’ means to read / write the elements using Fortran-like index order; keep the same as matlab

    sample['IU_inInd'] = Variable(th.from_numpy(raw_data['IU_inInd'].astype(np.int32)-1).cuda(), requires_grad=False) # index start from 1 in matlab, from 0 in python
    sample['IU_neighInd'] = Variable(th.from_numpy(raw_data['IU_neighInd'].astype(np.int32)-1).cuda(), requires_grad=False) # index start from 1 in matlab, from 0 in python
    sample['IU_flows'] = Variable(th.from_numpy(raw_data['IU_flows'].astype(np.float64)).cuda(), requires_grad=False)

    sample['LOC_inInd'] = Variable(th.from_numpy(raw_data['LOC_inInd'].astype(np.int32)-1).cuda(), requires_grad=False)
    sample['LOC_flows'] = Variable(th.from_numpy(raw_data['LOC_flows'].astype(np.float64)).cuda(), requires_grad=False)
    sample['matte'] = matte
    sample['trimap'] = trimap

    if self.transform is not None:
      sample = self.transform(sample)

    end = time.time()
    log.debug("load sample {:.2f}s/im".format((end-start)))
    return sample

class ToTensor(object):
  """Convert sample ndarrays to tensors."""
  def __call__(self, sample):
    xformed = {}
    for k in sample.keys():
      if type(sample[k]) == np.ndarray:
        xformed[k] = th.from_numpy(sample[k])
      else:
        xformed[k] = sample[k]
    return xformed
