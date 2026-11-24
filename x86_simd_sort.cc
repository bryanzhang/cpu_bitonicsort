// Baseline: Intel x86-simd-sort (https://github.com/intel/x86-simd-sort),
// header-only static API, single-threaded. The compile flags choose the
// AVX-512 or AVX2 implementation; see run_x86_simd_sort*.sh.
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "x86simdsort-static-incl.h"

#include "check_sorted.h"

constexpr int kSize = (1 << 13);
constexpr int kTries = 1000;

double test(std::vector<int>& arr) {
        for (int i = 0; i < kSize; ++i) {
                arr[i] = (rand() % 0x3ff);
        }
        sort_check.expect(arr.data(), kSize);

        auto start = std::chrono::high_resolution_clock::now();
        x86simdsortStatic::qsort(arr.data(), kSize);
        auto end = std::chrono::high_resolution_clock::now();
        sort_check.verify(arr.data(), kSize);
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count() * 1000;
}

int main() {
        srand(100);
        std::vector<int> arr(kSize);
        double millis = 0.0;
        for (int i = 0; i < kTries; ++i) {
                millis += test(arr);
        }
        std::cout << "n=" << kSize << ", average time: " << millis / kTries << " ms." << std::endl;
        return sort_check.report();
}
