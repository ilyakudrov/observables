GCC=g++
FLAGS=-Ofast -fopenmp -march=native -mtune=native -std=c++17
ARCH=test
OBJFILES=read_data_$(ARCH).o jackknife_$(ARCH).o
TARGETS=gluon_propagator

$(TARGETS): %: %.cpp $(OBJFILES)
	$(GCC) $@.cpp $(OBJFILES) -std=c++17 -fopenmp -o $@_$(ARCH)

$(OBJFILES): %_$(ARCH).o: %.cpp
	$(GCC) $(FLAGS) -c $< -o $@

clean:
	rm -rf *.o