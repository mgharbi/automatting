from torch.utils.ffi import create_extension

ffi = create_extension(
  name='_ext.sparse',
  package=False,
  headers='src/sparse.h',
  define_macros=[('WITH_CUDA', None)],
  sources=['src/sparse.cpp'],
  relative_to=__file__,
  with_cuda=True
)

if __name__ == '__main__':
  ffi.build()
