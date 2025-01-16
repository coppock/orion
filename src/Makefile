CUDA_TOOLKIT = /usr/local/cuda

all:
	cd cuda_capture && $(MAKE) CUDA_TOOLKIT=$(CUDA_TOOLKIT)
	cd scheduler && $(MAKE) CUDA_TOOLKIT=$(CUDA_TOOLKIT)

clean:
	cd cuda_capture && $(MAKE) clean
	cd scheduler && $(MAKE) clean
