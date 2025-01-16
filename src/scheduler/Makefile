CUDA_TOOLKIT = /usr/local/cuda

CXXFLAGS = -g -O3 -fPIC -shared -DDEBUG=1 -I$(CUDA_TOOLKIT)/include
LDFLAGS = -L$(CUDA_TOOLKIT)/lib64
LDLIBS = -lcudnn -lcudart -lcublas

scheduler_eval.so: scheduler_eval.o utils_sched.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o scheduler_eval.so utils_sched.o \
	    scheduler_eval.o $(LDLIBS)

scheduler_eval.o: scheduler.h
scheduler.h: utils_sched.h
utils_sched.o: utils_sched.h
utils_sched.h: ../system_utils.h

clean:
	rm -rf *.o
