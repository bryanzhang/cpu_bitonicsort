#! /bin/bash

# Baseline: Intel x86-simd-sort v7.0, AVX-512 implementation.
XSS_DIR=${XSS_DIR:-third_party/x86-simd-sort}
if [ ! -d "$XSS_DIR" ]; then
        git clone -q --depth 1 --branch v7.0 https://github.com/intel/x86-simd-sort.git "$XSS_DIR" || exit 1
fi
g++ -O3 -std=c++17 x86_simd_sort.cc -I"$XSS_DIR/src" -mavx512f -mavx512dq -mavx512vl && ./a.out
