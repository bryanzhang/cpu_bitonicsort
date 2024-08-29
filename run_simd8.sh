#! /bin/bash

g++ -O3 -std=c++14 5_simd8.cc -g -mavx2 && ./a.out
