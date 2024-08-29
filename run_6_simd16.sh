#! /bin/bash

g++ -O3 -std=c++14 6_simd16.cc -g -mavx512f && ./a.out
