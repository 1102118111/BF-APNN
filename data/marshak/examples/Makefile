INCDIR = ../
INCDIR1 = /home/susan/Install_Software/include

HEADER = $(wildcard $(INCDIR)/*.h)
HEADER1 = $(wildcard $(INCDIR1)/*.h)

BIN = $(patsubst %.cpp, %, $(wildcard *.cpp))

CXXFLAGS = -I$(INCDIR) -I$(INCDIR1) 

LDFLAGS = -L/home/susan/Install_Software/lib -std=c++11  -lm -lgsl -lgslcblas  #-llapack


ifeq (x$(debug), x)
	CXXFLAGS += -O3 -march=native -fopenmp -mavx2
else
	CXXFLAGS += -g -DDEBUG
endif

#ifneq (x$(mpi), x)
#CXX = mpicxx

#endif

all: $(BIN)
$(BIN): % : %.cpp $(HEADER) $(HEADER1)
	$(CXX) -o $@ $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS)

.PHONY: clean
clean:
	-rm $(BIN)


