cmake_minimum_required(VERSION 3.23)
project(Pn_gray_1D)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Pn_gray_1D
        examples/MarshakWave.cpp
        examples/plot_rad.m
        examples/test1.cpp
        FVM1D.h
        FVM1D.template.h
        Mesh1DED.h
        Pn_slab.h
        Solution.h
        Vector.h)
