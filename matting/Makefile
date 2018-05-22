TORCH_INC ?= `python3 -c 'import torch.utils.ffi as ffi; print("-I"+" -I".join(ffi._setup_wrapper(True)[1]))'`
NVCC = nvcc -std c++11 #-g -G # -pg
NVFLAGS = -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I$(TF_INC) -I$(SRC_DIR) \
					-gencode=arch=compute_30,code=\"sm_30,compute_30\" \
					-expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ftz=true --ptxas-options=-v -lineinfo

all: cffi

cffi: build/kernels.so
	python3 build.py

build/kernels.so: src/sparse_kernel.cu build src/sparse_kernel.h
	$(NVCC) -c  $< -o $@ $(NVFLAGS) $(TORCH_INC)

build:
	mkdir -p $@

clean:
	rm -rf build _ext
