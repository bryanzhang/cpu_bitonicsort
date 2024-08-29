#! /bin/bash

g++ -O3 -std=c++14 7_noqsort.cc -g -mavx512f && ./a.out
