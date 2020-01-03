cmake_minimum_required(VERSION 3.1)
project(miyuki.math)

set(CMAKE_CXX_STANDARD 17)

include_directories(include/)
file(GLOB vecSrc include/miyuki.math/*.*)

add_executable(simple-test tests/simple-test.cpp)