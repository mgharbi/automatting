import os
import re
import numpy as np
import scipy.io

import torch as th
from torch.utils.data import Dataset

# import matting.sparse as sp
import scipy.sparse as sp


class MattingDataset(Dataset):
  """"""

  def __init__(self, root_dir, transform=None):
    super(MattingDataset, self).__init__()
    self.root_dir = root_dir
    self.transform = transform
    file_list = os.listdir(self.root_dir)
    data_regex = re.compile(r".*.mat$")
    self.file_list = [f for f in file_list if data_regex.match(f)]

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, idx):
    data_file = os.path.join(self.root_dir, self.file_list[idx])

    print data_file

    data = scipy.io.loadmat(data_file)["IFMdata"]

    CM_inInd    = data['CM_inInd'][0][0].astype(np.int64)  # NOTE(mgharbi): this where saved as double
    CM_neighInd = data['CM_neighInd'][0][0].astype(np.int64)
    CM_flows    = data['CM_flows'][0][0].astype(np.float32)

    LOC_inInd    = data['LOC_inInd'][0][0].astype(np.int64)
    # LOC_flowRows = data['LOC_flowRows'][0][0]
    # LOC_flowCols = data['LOC_flowCols'][0][0]
    LOC_flows    = data['LOC_flows'][0][0].astype(np.float32)

    IU_inInd    = data['IU_inInd'][0][0].astype(np.int64)
    IU_neighInd = data['IU_neighInd'][0][0].astype(np.int64)
    IU_flows    = data['IU_flows'][0][0].astype(np.float32)

    kToU = data['kToU'][0][0].astype(np.float32)
    kToUconf = np.ravel(data['kToUconf'][0][0]).astype(np.float32)

    known = data['known'][0][0].astype(np.float32).ravel()

    h, w = kToU.shape
    N = h*w

    kToU = np.ravel(kToU)

    # Convert indices from matlab to numpy format
    CM_inInd     = self.convert_index(CM_inInd, h, w)
    CM_neighInd  = self.convert_index(CM_neighInd, h, w)
    LOC_inInd    = self.convert_index(LOC_inInd, h, w)
    # LOC_flowRows = self.convert_index(LOC_flowRows, h, w)
    # LOC_flowCols = self.convert_index(LOC_flowCols, h, w)
    IU_inInd     = self.convert_index(IU_inInd, h, w)
    IU_neighInd  = self.convert_index(IU_neighInd, h, w)

    # TODO(mgharbi): most of this pre-processing should be done once, off-line

    Wcm = self.color_mixture(N, CM_inInd, CM_neighInd, CM_flows)
    sample = {
        "Wcm_row": Wcm.row,
        "Wcm_col": Wcm.col,
        "Wcm_data": Wcm.data,
        "LOC_inInd": LOC_inInd,
        # "LOC_flowRows": LOC_flowRows,
        # "LOC_flowCols": LOC_flowCols,
        "LOC_flows": LOC_flows,
        "IU_inInd": IU_inInd,
        "IU_neighInd": IU_neighInd,
        "IU_flows": IU_flows,
        "kToUconf": kToUconf,
        "known": known,
        "kToU": kToU,
        "height": h,
        "width": w,
    }

    if self.transform is not None:
      sample = self.transform(sample)

    return sample

  def convert_index(self, old, h, w):
    idx = np.unravel_index(old, [w, h])
    new = np.ravel_multi_index((idx[1], idx[0]), (h, w)).astype(np.int32)
    return new

  def color_mixture(self, N, inInd, neighInd, flows):
    row_idx = np.tile(inInd, (1, flows.shape[1]))
    col_idx = neighInd
    Wcm = sp.coo_matrix((np.ravel(flows), (np.ravel(row_idx), np.ravel(col_idx))), shape=(N, N))
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
