
def color_mixture_laplacian(N, inInd, neighInd, flows, weights):
  """ """

  row_idx = np.tile(inInd, (1, flows.shape[1]))
  col_idx = neighInd

  Wcm = sp.coo_matrix((np.ravel(flows), (np.ravel(row_idx), np.ravel(col_idx))), shape=(N, N))
  Wcm = sp.spdiags(np.ravel(weights), 0, N, N).dot(Wcm)
  Lcm = sp.spdiags(np.ravel(np.sum(Wcm, axis=1)), 0, N, N) - Wcm
  Lcm = (Lcm.T).dot(Lcm)

  return Lcm
