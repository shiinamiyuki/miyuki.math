cmake_minimum_required(VERSION 3.1)
project(miyuki.vectorize)

set(CMAKE_CXX_STANDARD 17)

include_directories(include/)
file(GLOB vecSrc include/miyuki.vectorize/*.*)

add_executable(simple-test tests/simple-test.cpp)