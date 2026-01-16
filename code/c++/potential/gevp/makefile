GCC=g++
FLAGS=-O3 -fopenmp -march=native -mtune=native -std=c++17
ARCH=test
OBJFILES=read_data_$(ARCH).o jackknife_$(ARCH).o
TARGETS=potential_gevp force_gevp
EIGH_PATH="/home/ilya/soft/source/eigen-master"

$(TARGETS): %: %.cpp $(OBJFILES)
	$(GCC) $@.cpp $(OBJFILES) -I $(EIGH_PATH) -std=c++17 -fopenmp -o $@_$(ARCH)

$(OBJFILES): %_$(ARCH).o: %.cpp
	$(GCC) $(FLAGS) -I $(EIGH_PATH) -c $< -o $@

clean:
	rm -rf *.o