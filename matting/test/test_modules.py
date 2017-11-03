import torch as th
from torch.autograd import Variable

import matting.modules as modules
import matplotlib.pyplot as plt

def test_alpha_gradient():
  f = modules.AlphaGradientNorm(blur_std=2)
  alpha = Variable(th.randn(1, 1, 30, 30))
  for i in range(15):
    alpha.data[:, :, i, i] = 5
  g = f(alpha)

  # plt.imshow(g.data[0, 0, ...].numpy())
  # plt.show()
