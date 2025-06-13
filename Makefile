# Makefile for Mantiuk Tone Mapping CUDA + CPU implementations

NVCC        = nvcc
CXX         = g++
CXXFLAGS    = -O3 -std=c++14
OPENCVFLAGS = $(shell pkg-config --cflags --libs opencv4)

TARGETS = mantiuk_shared mantiuk_cpu

all: $(TARGETS)

mantiuk_shared: mantiuk.cu
	$(NVCC) $(CXXFLAGS) $< -o $@ $(OPENCVFLAGS)

mantiuk_cpu: cpu_version.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(OPENCVFLAGS)

clean:
	rm -f $(TARGETS)