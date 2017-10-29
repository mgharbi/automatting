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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MattingDataset(Dataset):
  """"""

  def __init__(self, root_dir, transform=None):
    super(MattingDataset, self).__init__()
    self.transform = transform

    self.root_dir = root_dir
    self.ifm_data_dir = os.path.join(root_dir, 'IFMdata')
    self.matte_dir = os.path.join(root_dir, 'alpha')
    self.images_dir = os.path.join(root_dir, 'images')
    # self.trimap_dir = os.path.join(root_dir, 'trimap1')

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
      self.files.append(f)
    fid.close()

    duration = time.time() - start

    log.info("Parsed dataset {} with {} samples in {:.2f}s".format(
      root_dir, len(self), duration))


  def image_path(self, f):
    return os.path.join(self.images_dir, f)

  def basename(self, f):
    fname = os.path.splitext(f)[0]
    basename = "_".join(fname.split("_")[:-1])
    return basename

  def ifm_path(self, f):
    return os.path.join(self.ifm_data_dir, os.path.splitext(f)[0]+".mat")

  def matte_path(self, f):
    return os.path.join(self.matte_dir, self.basename(f)+".jpg")

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    start = time.time()
    fname = self.files[idx]

    matte = skimage.io.imread(self.matte_path(fname)).astype(np.float32)/255.0
    image = skimage.io.imread(self.image_path(fname)).astype(np.float32)/255.0
    # matte = matte.transpose([2, 0, 1])
    image = image.transpose([2, 0, 1])

    data = scipy.io.loadmat(self.ifm_path(fname))["IFMdata"]

    CM_inInd    = data['CM_inInd'][0][0].astype(np.int64)  # NOTE(mgharbi): these are saved as floats
    CM_neighInd = data['CM_neighInd'][0][0].astype(np.int64)
    CM_flows    = data['CM_flows'][0][0]

    LOC_inInd    = data['LOC_inInd'][0][0].astype(np.int64)
    LOC_flows    = data['LOC_flows'][0][0]

    IU_inInd    = data['IU_inInd'][0][0].astype(np.int64)
    IU_neighInd = data['IU_neighInd'][0][0].astype(np.int64)
    IU_flows    = data['IU_flows'][0][0]

    kToU = data['kToU'][0][0]
    kToUconf = np.ravel(data['kToUconf'][0][0])

    known = data['known'][0][0].ravel()

    h, w = kToU.shape
    N = h*w

    kToU = np.ravel(kToU)

    # Convert indices from matlab to numpy format
    CM_inInd     = self.convert_index(CM_inInd, h, w)
    CM_neighInd  = self.convert_index(CM_neighInd, h, w)
    LOC_inInd    = self.convert_index(LOC_inInd, h, w)
    IU_inInd     = self.convert_index(IU_inInd, h, w)
    IU_neighInd  = self.convert_index(IU_neighInd, h, w)

    Wcm = self.color_mixture(N, CM_inInd, CM_neighInd, CM_flows)
    sample = {
        "Wcm_row": np.squeeze(Wcm.row),
        "Wcm_col": np.squeeze(Wcm.col),
        "Wcm_data": np.squeeze(Wcm.data),
        "LOC_inInd": LOC_inInd,
        "LOC_flows": LOC_flows,
        "IU_inInd": IU_inInd,
        "IU_neighInd": IU_neighInd,
        "IU_flows": IU_flows,
        "kToUconf": kToUconf,
        "known": known,
        "kToU": kToU,
        "height": h,
        "width": w,
        "image": image,
        "matte": matte,
    }

    if self.transform is not None:
      sample = self.transform(sample)

    end = time.time()
    log.info("load sample {:.2f}s/im".format((end-start)))
    return sample

  def convert_index(self, old, h, w):
    idx = np.unravel_index(old, [w, h])
    new = np.ravel_multi_index((idx[1], idx[0]), (h, w)).astype(np.int32)
    return new

  def color_mixture(self, N, inInd, neighInd, flows):
    row_idx = np.tile(inInd, (1, flows.shape[1]))
    col_idx = neighInd
    Wcm = sp.coo_matrix(
        (np.ravel(flows), (np.ravel(row_idx), np.ravel(col_idx))), shape=(N, N))
    return Wcm


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
