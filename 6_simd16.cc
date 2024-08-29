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
        for (int* p = arr, *q = arr + 4096; p < arr + 4096; p += 16, q += 16) {
                __m512i a = _mm512_load_si512((__m512i*)p);
                __m512i b = _mm512_load_si512((__m512i*)q);
                __m512i min_result = _mm512_min_epi32(a, b);
                __m512i max_result = _mm512_max_epi32(a, b);
                _mm512_store_si512((__m512i*)p, min_result);
                _mm512_store_si512((__m512i*)q, max_result);
        }

        // stride = 2048
        for (int* p = arr, *q = arr + 2048, *sect_end = arr + 2048; q < arr + n; sect_end += 2048, p += 2048, q += 2048) {
// #pragma unroll
                for (; p < sect_end; p += 16, q += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)q);
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, min_result);
                        _mm512_store_si512((__m512i*)q, max_result);
                }
        }

        // stride = 1024
        for (int* p = arr, *q = arr + 1024, *sect_end = arr + 1024; q < arr + n; sect_end += 1024, p += 1024, q += 1024) {
// #pragma unroll
                for (; p < sect_end; p += 16, q += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)q);
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, min_result);
                        _mm512_store_si512((__m512i*)q, max_result);
                }
        }

        // stride = 512
        for (int* p = arr, *q = arr + 512, *sect_end = arr + 512; q < arr + n; sect_end += 512, p += 512, q += 512) {
                for (; p < sect_end; p += 16, q += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)q);
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, min_result);
                        _mm512_store_si512((__m512i*)q, max_result);
                }
        }

        // stride = 256
        for (int* p = arr, *q = arr + 256, *sect_end = arr + 256; q < arr + n; sect_end += 256, p += 256, q += 256) {
                for (; p < sect_end; p += 16, q += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)q);
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, min_result);
                        _mm512_store_si512((__m512i*)q, max_result);
                }
        }

        // stride = 128
        for (int* p = arr, *q = arr + 128, *sect_end = arr + 128; q < arr + n; sect_end += 128, p += 128, q += 128) {
                for (; p < sect_end; p += 16, q += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)q);
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, min_result);
                        _mm512_store_si512((__m512i*)q, max_result);
                }
        }

        // stride = 64
        for (int* p = arr; p < arr + n; p += 128) {
                __m512i a = _mm512_load_si512((__m512i*)p);
                __m512i b = _mm512_load_si512((__m512i*)(p + 64));
                __m512i min_result = _mm512_min_epi32(a, b);
                __m512i max_result = _mm512_max_epi32(a, b);
                _mm512_store_si512((__m512i*)p, min_result);
                _mm512_store_si512((__m512i*)(p + 64), max_result);

                __m512i c = _mm512_load_si512((__m512i*)(p + 16));
                __m512i d = _mm512_load_si512((__m512i*)(p + 80));
                __m512i min_result1 = _mm512_min_epi32(c, d);
                __m512i max_result1 = _mm512_max_epi32(c, d);
                _mm512_store_si512((__m512i*)(p + 16), min_result1);
                _mm512_store_si512((__m512i*)(p + 80), max_result1);

                __m512i e = _mm512_load_si512((__m512i*)(p + 32));
                __m512i f = _mm512_load_si512((__m512i*)(p + 96));
                __m512i min_result2 = _mm512_min_epi32(e, f);
                __m512i max_result2 = _mm512_max_epi32(e, f);
                _mm512_store_si512((__m512i*)(p + 32), min_result2);
                _mm512_store_si512((__m512i*)(p + 96), max_result2);

                __m512i g = _mm512_load_si512((__m512i*)(p + 48));
                __m512i h = _mm512_load_si512((__m512i*)(p + 112));
                __m512i min_result3 = _mm512_min_epi32(g, h);
                __m512i max_result3 = _mm512_max_epi32(g, h);
                _mm512_store_si512((__m512i*)(p + 48), min_result3);
                _mm512_store_si512((__m512i*)(p + 112), max_result3);
        }

        // stride = 32
        for (int* p = arr; p < arr + n; p += 64) {
                __m512i a = _mm512_load_si512((__m512i*)p);
                __m512i b = _mm512_load_si512((__m512i*)(p + 32));
                __m512i min_result = _mm512_min_epi32(a, b);
                __m512i max_result = _mm512_max_epi32(a, b);
                _mm512_store_si512((__m512i*)p, min_result);
                _mm512_store_si512((__m512i*)(p + 32), max_result);

                __m512i c = _mm512_load_si512((__m512i*)(p + 16));
                __m512i d = _mm512_load_si512((__m512i*)(p + 48));
                __m512i min_result1 = _mm512_min_epi32(c, d);
                __m512i max_result1 = _mm512_max_epi32(c, d);
                _mm512_store_si512((__m512i*)(p + 16), min_result1);
                _mm512_store_si512((__m512i*)(p + 48), max_result1);
        }

        // stride = 16
        for (int* p = arr; p < arr + n; p += 32) {
                __m512i a = _mm512_load_si512((__m512i*)p);
                __m512i b = _mm512_load_si512((__m512i*)(p + 16));
                __m512i min_result = _mm512_min_epi32(a, b);
                __m512i max_result = _mm512_max_epi32(a, b);
                _mm512_store_si512((__m512i*)p, min_result);
                _mm512_store_si512((__m512i*)(p + 16), max_result);
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
        int* arr = (int*)aligned_alloc(sizeof(int) * 32, size * sizeof(int));
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
