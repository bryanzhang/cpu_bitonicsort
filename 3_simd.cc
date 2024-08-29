#include <chrono>
#include <cstdlib>
#include <iostream>

#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

constexpr int size = (1 << 13);
// constexpr int size = (1 << 4);
constexpr int tries = 1000;

inline void compareAndSwap(int* p, int* q) {
        if (*p > *q) {
                int tmp = *p;
                *p = *q;
                *q = tmp;
        }
}

// arr中已经是双调数列.
// NOTE: 数组需要padding到8的倍数
void bitonic_sort(int* arr, int n) {
        if (n == 1) { [[unlikely]]
                return;
        }

        // NOTE: 并到后面的循环中反而执行变慢
        for (int* p = arr, *q = arr + (n >> 1); p < arr + (n >> 1); ++p, ++q) {
                compareAndSwap(p, q);
        }

#pragma unroll
        for (int stride = (n >> 2); stride >= 8; stride >>= 1) {
                for (int* p = arr, *q = arr + stride, *sect_end = arr + stride; q < arr + n; sect_end += stride, p += stride, q += stride) {
                        for (; p < sect_end; ++p, ++q) {
                                compareAndSwap(p, q);
                        }
                }
        }

        for (int* p = arr; p < arr + n; p += 8) {
                __m128i a = _mm_load_si128((__m128i*)p);
                __m128i b = _mm_load_si128((__m128i*)(p + 4));
                __m128i min_result = _mm_min_epi32(a, b);
                __m128i max_result = _mm_max_epi32(a, b);
                _mm_store_si128((__m128i*)p, min_result);
                _mm_store_si128((__m128i*)(p + 4), max_result);
        }
}

int asc_compare(const void* a, const void* b) {
        int arg1 = *(const int *)a;
        int arg2 = *(const int *)b;
        return arg1 - arg2;
}

int desc_compare(const void* a, const void* b) {
        int arg1 = *(const int *)a;
        int arg2 = *(const int *)b;
        return arg2 - arg1;
}

double test(int* arr) {
        for (int i = 0; i < size; ++i) {
                arr[i] = (rand() % 0x3ff);
        }

        auto start = chrono::high_resolution_clock::now();
        // make bitnoic
        qsort(arr, (size >> 1), sizeof(int), asc_compare);
        qsort(arr + (size >> 1), size - (size >> 1), sizeof(int), desc_compare);
        bitonic_sort(arr, size);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        return elapsed.count() * 1000;
}

int main() {
        srand(100);
        int* arr = new int[size];
        double millis = 0.0;
        for (int i = 0; i < tries; ++i) {
                millis += test(arr);
        }
        // for (int i = 0; i < size; ++i) {
        //      cout << arr[i] << ", ";
        // }
        cout << endl;
        cout << "n=" << size << ", average time: " << millis / tries << " ms." << endl;
        delete[] arr;
        return 0;
}
