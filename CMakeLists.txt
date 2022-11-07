cmake_minimum_required(VERSION 3.22)
project(para)

set(CMAKE_CXX_STANDARD 23)

add_executable(para main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
