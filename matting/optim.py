import logging

import torch as th

import matting.sparse as sp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def cg(A, b, x0, steps=1, thresh=1e-5, verbose=False):
  r = b - A.matmul(x0)
  p = r.clone()
  x = x0.clone()
  res_old = r.dot(r)
  err = -1

  for k in range(steps):
    Ap = A.matmul(p)
    alpha = res_old / p.dot(Ap)
    x = x +  alpha*p
    r = r - alpha*Ap
    res_new = r.dot(r)
    err = th.sqrt(res_new).data[0]
    if (err < thresh):
      if verbose:
        log.info("CG converged with residual {}.".format(err))
      break
    if verbose:
      log.info("CG step {} / {}, residual = {:g}".format(k+1, steps, err))
    p = r + res_new/res_old*p
    res_old = res_new
  return x, err


def sparse_cg(A, b, x0, steps=1, thresh=1e-4, verbose=False):
  r = b - sp.spmv(A, x0)
  p = r.clone()
  x = x0.clone()
  res_old = r.dot(r)
  err = -1

  for k in range(steps):
    Ap = sp.spmv(A, p)
    alpha = res_old / p.dot(Ap)
    x = x +  alpha*p
    r = r - alpha*Ap
    res_new = r.dot(r)
    err = th.sqrt(res_new).data[0]
    if (err < thresh):
      if verbose:
        log.info("CG converged with residual {}.".format(err))
      break
    if verbose:
      log.info("CG step {} / {}, residual = {:g}".format(k+1, steps, err))
    p = r + res_new/res_old*p
    res_old = res_new
  return x, err, k+1


def line_search(A, b, x0, p):
"""Function to perform 'line (magnitude) search'
Parameters
----------
A: Sparse Object from matting.sparse, consisting of torch Variable
    matrix of quadratic function
b: Sparse Object from matting.sparse, consisting of torch Variable
    vector of quadratic function
x0: Sparse Object from matting.sparse, consisting of torch Variable
    starting point of line line_search
p: Sparse Object from matting.sparse, consisting of torch Variable
    conjugate direction

Returns
-------
output: float32
    optimal magnitude for conjugate direction
"""
    candidate = [0.1**i for i in range(1,5)]
    err_logs = []
    for alpha in candidate:
        tmp = x0 + alpha*p
        err = b - sp.spmv(A, tmp)
        err = err.dot(err)
        err = th.sqrt(err).data[0]
        err_logs.append(err)
    ind = np.argmin(err_logs)
    return candidate[ind]


def sparse_ncg_prls(A, b, x0, steps=1, thresh=1e-4, verbose=False):
"""Function to perform non conjugate gradients opimisation, use Polak–Ribière method compute beta.
Parameters
----------
A: Sparse Object from matting.sparse, consisting of torch Variable
    matrix of quadratic function
b: Sparse Object from matting.sparse, consisting of torch Variable
    vector of quadratic function
x0: Sparse Object from matting.sparse, consisting of torch Variable
    starting point of line line_search
steps: int
    max. steps of optimisation
thresh: float
    threshold for convergence

Returns
-------
x: Sparse Object from matting.sparse, consisting of torch Variable
    optimisation results
err: float
    final error
k: int
    total steps
"""
  r = b - sp.spmv(A, x0) # direction of negative gradient, i.e. delta_x
  p = r.clone() # direction of subsequent conjugate direction, i.e. s
  x = x0.clone()
  res_old = r.dot(r)
  err = -1
  result_logs = []
  err_logs = []


  for k in range(steps):

    r_old = r
    r = b - sp.spmv(A, x)
    beta = r.dot(r - r_old) / r_old.dot(r_old)
    print('beta', beta.data[0])
    p = r + beta*p
    alpha = line_search(A, b, x, p)
    x = x + alpha*p

    result_logs.append(x)

    res_err = b - sp.spmv(A, x)
    res_err = res_err.dot(res_err)
    err = th.sqrt(res_err).data[0]
    err_logs.append(err)


    if (err < thresh):
      if verbose:
        log.info("CG converged with residual {}.".format(err))
      break
    if verbose:
      log.info("CG step {} / {}, residual = {:g}".format(k+1, steps, err))

  smallest_ind = np.argmin(err_logs)
  print('total length of logs', len(result_logs))
  print('smallest index', smallest_ind)

  return result_logs[smallest_ind], err_logs[smallest_ind], k+1


def sparse_cg_ib(A, b, x0, gt, steps=1, thresh=1e-4, verbose=False):
"""Function to perform non conjugate gradients opimisation, use intermediate best results
Parameters
----------
A: Sparse Object from matting.sparse, consisting of torch Variable
    matrix of quadratic function
b: Sparse Object from matting.sparse, consisting of torch Variable
    vector of quadratic function
x0: Sparse Object from matting.sparse, consisting of torch Variable
    starting point of line line_search
gt: Sparse Object from matting.sparse, consisting of torch Variable
    groud truth alpha
steps: int
    max. steps of optimisation
thresh: float
    threshold for convergence

Returns
-------
x: Sparse Object from matting.sparse, consisting of torch Variable
    optimisation results
err: float
    final error
k: int
    total steps
"""
  A_mat = A.to_dense()
  A_mat = np.asarray(A_mat)
  r = b - sp.spmv(A, x0)
  p = r.clone()
  x = x0.clone()
  res_old = r.dot(r)
  err = -1
  result_logs = []
  err_logs = []


  err_gt = np.mean((x0.cpu().data.numpy()-gt.cpu().data.numpy())**2)
  err_gt_logs = []

  for k in range(steps):
    Ap = sp.spmv(A, p)
    alpha = res_old / p.dot(Ap)
    x = x +  alpha*p
    r = r - alpha*Ap
    res_new = r.dot(r)
    err = th.sqrt(res_new).data[0]
    result_logs.append(x)

    res_err = b.cpu().data.numpy() - np.matmul(A_mat, x.cpu().data.numpy())
    err_logs.append(np.dot(res_err, res_err)**(0.5))

    err_gt = np.mean((x.cpu().data.numpy()-gt.cpu().data.numpy())**2)
    err_gt_logs.append(err_gt)

    if (err < thresh):
      if verbose:
        log.info("CG converged with residual {}.".format(err))
      break
    if verbose:
      log.info("CG step {} / {}, residual = {:g}".format(k+1, steps, err))
    p = r + res_new/res_old*p
    res_old = res_new

  smallest_ind = np.argmin(err_logs)
  sml_gt_ind = np.argmin(err_gt_logs)
  print('total length of logs', len(result_logs))
  print('smallest index', smallest_ind)
  print('smallest gt index', sml_gt_ind)

  return result_logs[smallest_ind], err_logs[smallest_ind], k+1
