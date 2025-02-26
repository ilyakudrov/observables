GCC=g++
FLAGS=-O3 -fopenmp -march=native -mtune=native -std=c++23
ARCH=test
OBJFILES=read_data_$(ARCH).o jackknife_$(ARCH).o make_observables_$(ARCH).o data_processing_$(ARCH).o
TARGETS=average_observables average_distribution average_observables_rings
DATAFRAME_PATH="/home/ilya/soft/source/DataFrame/include"
DATAFRAME_OBJ=/home/ilya/soft/source/DataFrame/obj/Linux.GCC64/DateTime.o

$(TARGETS): %: %.cpp $(OBJFILES) visitors.h
	$(GCC) $@.cpp $(OBJFILES) -I$(DATAFRAME_PATH) -std=c++23 -fopenmp -o $@_$(ARCH)

$(OBJFILES): %_$(ARCH).o: %.cpp visitors.h
	$(GCC) $(FLAGS) -I$(DATAFRAME_PATH) -c $< -o $@

clean:
	rm -rf *.o