CUDA_TOOLKIT = /usr/local/cuda

CXXFLAGS = -O3 -fPIC -shared -I$(CUDA_TOOLKIT)/include
LDFLAGS = -L$(CUDA_TOOLKIT)/lib64
LDLIBS = -lcudart

libinttemp.so: utils_interc.o intercept_cudnn.o intercept_cublas.o \
    intercept_temp.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o libinttemp.so utils_interc.o \
	    intercept_cudnn.o intercept_cublas.o intercept_temp.o $(LDLIBS)

intercept_cublas.o: intercept_temp.h
intercept_cudnn.o: intercept_temp.h
intercept_temp.o: intercept_temp.h
intercept_h: ../system_utils.h

clean:
	rm -rf *.o
