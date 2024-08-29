#! /bin/bash

g++ -O3 -std=c++14 3_simd.cc -mavx2 -std=c++14 && ./a.out
