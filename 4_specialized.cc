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
void bitonic_sort8192(int* arr, int n) {
        // stride = 4096
        for (int* p = arr, *q = arr + 4096; p < arr + 4096; p += 4, q += 4) {
                __m128i a = _mm_load_si128((__m128i*)p);
                __m128i b = _mm_load_si128((__m128i*)q);
                __m128i min_result = _mm_min_epi32(a, b);
                __m128i max_result = _mm_max_epi32(a, b);
                _mm_store_si128((__m128i*)p, min_result);
                _mm_store_si128((__m128i*)q, max_result);
        }

        // stride = 2048
        for (int* p = arr, *q = arr + 2048, *sect_end = arr + 2048; q < arr + n; sect_end += 2048, p += 2048, q += 2048) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 1024
        for (int* p = arr, *q = arr + 1024, *sect_end = arr + 1024; q < arr + n; sect_end += 1024, p += 1024, q += 1024) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 512
        for (int* p = arr, *q = arr + 512, *sect_end = arr + 512; q < arr + n; sect_end += 512, p += 512, q += 512) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 256
        for (int* p = arr, *q = arr + 256, *sect_end = arr + 256; q < arr + n; sect_end += 256, p += 256, q += 256) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 128
        for (int* p = arr, *q = arr + 128, *sect_end = arr + 128; q < arr + n; sect_end += 128, p += 128, q += 128) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 64
        for (int* p = arr, *q = arr + 64, *sect_end = arr + 64; q < arr + n; sect_end += 64, p += 64, q += 64) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 32
        for (int* p = arr, *q = arr + 32, *sect_end = arr + 32; q < arr + n; sect_end += 32, p += 32, q += 32) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 16
        for (int* p = arr, *q = arr + 16, *sect_end = arr + 16; q < arr + n; sect_end += 16, p += 16, q += 16) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 8
        for (int* p = arr, *q = arr + 8, *sect_end = arr + 8; q < arr + n; sect_end += 8, p += 8, q += 8) {
// #pragma unroll
                for (; p < sect_end; p += 4, q += 4) {
                        __m128i a = _mm_load_si128((__m128i*)p);
                        __m128i b = _mm_load_si128((__m128i*)q);
                        __m128i min_result = _mm_min_epi32(a, b);
                        __m128i max_result = _mm_max_epi32(a, b);
                        _mm_store_si128((__m128i*)p, min_result);
                        _mm_store_si128((__m128i*)q, max_result);
                }
        }

        // stride = 4
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
        bitonic_sort8192(arr, size);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        return elapsed.count() * 1000;
}

int main() {
        srand(100);
        int* arr = (int*)aligned_alloc(sizeof(int) * 8, size * sizeof(int));
        cout << "arr pointer: " << arr << endl;
        double millis = 0.0;
        for (int i = 0; i < tries; ++i) {
                millis += test(arr);
        }
        // for (int i = 0; i < size; ++i) {
        //      cout << arr[i] << ", ";
        // }
        cout << endl;
        cout << "n=" << size << ", average time: " << millis / tries << " ms." << endl;
        free(arr);
        return 0;
}

