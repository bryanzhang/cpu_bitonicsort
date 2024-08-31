#include <chrono>
#include <cstdlib>
#include <iostream>

#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

constexpr int size = (1 << 13);
// constexpr int size = (1 << 4);
constexpr int tries = 1000;

// arr中已经是双调数列.
// n为2的幂，且小于等于8192, 大于等于32.
template <int n>
void bitonic_sort(int* arr) {
        if (n == 8192) {
                // stride = 4096
                for (int* p = arr; p < arr + 4096; p += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)(p + 4096));
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, min_result);
                        _mm512_store_si512((__m512i*)(p + 4096), max_result);
                }
        }

        // stride = 2048
        if (n >= 4096) {
                for (int* p = arr, *q = arr + 2048, *sect_end = arr + 2048; q < arr + n; sect_end += 2048, p += 2048, q += 2048) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, min_result);
                                _mm512_store_si512((__m512i*)q, max_result);
                        }
                }
        }

        // stride = 1024
        if (n >= 2048) {
                for (int* p = arr, *q = arr + 1024, *sect_end = arr + 1024; q < arr + n; sect_end += 1024, p += 1024, q += 1024) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, min_result);
                                _mm512_store_si512((__m512i*)q, max_result);
                        }
                }
        }

        // stride = 512
        if (n >= 1024) {
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
        }

        // stride = 256
        if (n >= 512) {
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
        }

        // stride = 128
        if (n >= 256) {
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
        }

        // stride = 64
        if (n >= 128) {
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
        }

        // stride = 32
        if (n >= 64) {
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

// arr中已经是双调数列.
// n为2的幂，且小于等于8192, 大于等于32.
template <int n>
void bitonic_sort_desc(int* arr) {
        if (n == 8192) {
                // stride = 4096
                for (int* p = arr; p < arr + 4096; p += 16) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)(p + 4096));
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, max_result);
                        _mm512_store_si512((__m512i*)(p + 4096), min_result);
                }
        }

        // stride = 2048
        if (n >= 4096) {
                for (int* p = arr, *q = arr + 2048, *sect_end = arr + 2048; q < arr + n; sect_end += 2048, p += 2048, q += 2048) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, max_result);
                                _mm512_store_si512((__m512i*)q, min_result);
                        }
                }
        }

        // stride = 1024
        if (n >= 2048) {
                for (int* p = arr, *q = arr + 1024, *sect_end = arr + 1024; q < arr + n; sect_end += 1024, p += 1024, q += 1024) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, max_result);
                                _mm512_store_si512((__m512i*)q, min_result);
                        }
                }
        }

        // stride = 512
        if (n >= 1024) {
                for (int* p = arr, *q = arr + 512, *sect_end = arr + 512; q < arr + n; sect_end += 512, p += 512, q += 512) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, max_result);
                                _mm512_store_si512((__m512i*)q, min_result);
                        }
                }
        }

        // stride = 256
        if (n >= 512) {
                for (int* p = arr, *q = arr + 256, *sect_end = arr + 256; q < arr + n; sect_end += 256, p += 256, q += 256) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, max_result);
                                _mm512_store_si512((__m512i*)q, min_result);
                        }
                }
        }

        // stride = 128
        if (n >= 256) {
                for (int* p = arr, *q = arr + 128, *sect_end = arr + 128; q < arr + n; sect_end += 128, p += 128, q += 128) {
                        for (; p < sect_end; p += 16, q += 16) {
                                __m512i a = _mm512_load_si512((__m512i*)p);
                                __m512i b = _mm512_load_si512((__m512i*)q);
                                __m512i min_result = _mm512_min_epi32(a, b);
                                __m512i max_result = _mm512_max_epi32(a, b);
                                _mm512_store_si512((__m512i*)p, max_result);
                                _mm512_store_si512((__m512i*)q, min_result);
                        }
                }
        }

        // stride = 64
        if (n >= 128) {
                for (int* p = arr; p < arr + n; p += 128) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)(p + 64));
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, max_result);
                        _mm512_store_si512((__m512i*)(p + 64), min_result);

                        __m512i c = _mm512_load_si512((__m512i*)(p + 16));
                        __m512i d = _mm512_load_si512((__m512i*)(p + 80));
                        __m512i min_result1 = _mm512_min_epi32(c, d);
                        __m512i max_result1 = _mm512_max_epi32(c, d);
                        _mm512_store_si512((__m512i*)(p + 16), max_result1);
                        _mm512_store_si512((__m512i*)(p + 80), min_result1);

                        __m512i e = _mm512_load_si512((__m512i*)(p + 32));
                        __m512i f = _mm512_load_si512((__m512i*)(p + 96));
                        __m512i min_result2 = _mm512_min_epi32(e, f);
                        __m512i max_result2 = _mm512_max_epi32(e, f);
                        _mm512_store_si512((__m512i*)(p + 32), max_result2);
                        _mm512_store_si512((__m512i*)(p + 96), min_result2);

                        __m512i g = _mm512_load_si512((__m512i*)(p + 48));
                        __m512i h = _mm512_load_si512((__m512i*)(p + 112));
                        __m512i min_result3 = _mm512_min_epi32(g, h);
                        __m512i max_result3 = _mm512_max_epi32(g, h);
                        _mm512_store_si512((__m512i*)(p + 48), max_result3);
                        _mm512_store_si512((__m512i*)(p + 112), min_result3);
                }
        }

        // stride = 32
        if (n >= 64) {
                for (int* p = arr; p < arr + n; p += 64) {
                        __m512i a = _mm512_load_si512((__m512i*)p);
                        __m512i b = _mm512_load_si512((__m512i*)(p + 32));
                        __m512i min_result = _mm512_min_epi32(a, b);
                        __m512i max_result = _mm512_max_epi32(a, b);
                        _mm512_store_si512((__m512i*)p, max_result);
                        _mm512_store_si512((__m512i*)(p + 32), min_result);

                        __m512i c = _mm512_load_si512((__m512i*)(p + 16));
                        __m512i d = _mm512_load_si512((__m512i*)(p + 48));
                        __m512i min_result1 = _mm512_min_epi32(c, d);
                        __m512i max_result1 = _mm512_max_epi32(c, d);
                        _mm512_store_si512((__m512i*)(p + 16), max_result1);
                        _mm512_store_si512((__m512i*)(p + 48), min_result1);
                }
        }

        // stride = 16
        for (int* p = arr; p < arr + n; p += 32) {
                __m512i a = _mm512_load_si512((__m512i*)p);
                __m512i b = _mm512_load_si512((__m512i*)(p + 16));
                __m512i min_result = _mm512_min_epi32(a, b);
                __m512i max_result = _mm512_max_epi32(a, b);
                _mm512_store_si512((__m512i*)p, max_result);
                _mm512_store_si512((__m512i*)(p + 16), min_result);
        }
}

inline void make_bitonic8192(int* arr) {
        // stride = 64,每个块内排序
        for (int* p = arr; p < arr + 8192; p += 64) {
                // 4*4 in-register sort.
                // 1.load
                __m256i row0 = _mm256_loadu_si256((__m256i*)p);
                __m256i row1 = _mm256_loadu_si256((__m256i*)(p + 8));
                __m256i row2 = _mm256_loadu_si256((__m256i*)(p + 16));
                __m256i row3 = _mm256_loadu_si256((__m256i*)(p + 24));
                __m256i row4 = _mm256_loadu_si256((__m256i*)(p + 32));
                __m256i row5 = _mm256_loadu_si256((__m256i*)(p + 40));
                __m256i row6 = _mm256_loadu_si256((__m256i*)(p + 48));
                __m256i row7 = _mm256_loadu_si256((__m256i*)(p + 56));

                // 2.分channel排序
                // round 1
                __m256i row0_0 = _mm256_min_epi32(row0, row1);
                __m256i row1_0 = _mm256_min_epi32(row2, row3);
                __m256i row2_0 = _mm256_min_epi32(row4, row5);
                __m256i row3_0 = _mm256_min_epi32(row6, row7);

                __m256i row4_0 = _mm256_max_epi32(row0, row1);
                __m256i row5_0 = _mm256_max_epi32(row2, row3);
                __m256i row6_0 = _mm256_max_epi32(row4, row5);
                __m256i row7_0 = _mm256_max_epi32(row6, row7);

                // round 2
                __m256i row0_1 = _mm256_min_epi32(row0_0, row1_0);
                __m256i row1_1 = _mm256_min_epi32(row2_0, row3_0);

                __m256i row2_1 = _mm256_max_epi32(row0_0, row1_0);
                __m256i row3_1 = _mm256_max_epi32(row2_0, row3_0);

                __m256i row4_1 = _mm256_min_epi32(row4_0, row5_0);
                __m256i row5_1 = _mm256_min_epi32(row6_0, row7_0);

                __m256i row6_1 = _mm256_max_epi32(row4_0, row5_0);
                __m256i row7_1 = _mm256_max_epi32(row6_0, row7_0);

                // round 3
                __m256i row0_2 = _mm256_min_epi32(row0_1, row1_1);
                __m256i row1_2 = _mm256_max_epi32(row0_1, row1_1);
                __m256i row2_2 = _mm256_min_epi32(row2_1, row3_1);
                __m256i row3_2 = _mm256_max_epi32(row2_1, row3_1);
                __m256i row4_2 = _mm256_min_epi32(row4_1, row5_1);
                __m256i row5_2 = _mm256_max_epi32(row4_1, row5_1);
                __m256i row6_2 = _mm256_min_epi32(row6_1, row7_1);
                __m256i row7_2 = _mm256_max_epi32(row6_1, row7_1);

                // 3.转置
                __m256i t0 = _mm256_unpacklo_epi32(row0_2, row4_2);
                __m256i t1 = _mm256_unpackhi_epi32(row0_2, row4_2);
                __m256i t2 = _mm256_unpacklo_epi32(row1_2, row5_2);
                __m256i t3 = _mm256_unpackhi_epi32(row1_2, row5_2);
                __m256i t4 = _mm256_unpacklo_epi32(row2_2, row6_2);
                __m256i t5 = _mm256_unpackhi_epi32(row2_2, row6_2);
                __m256i t6 = _mm256_unpacklo_epi32(row3_2, row7_2);
                __m256i t7 = _mm256_unpackhi_epi32(row3_2, row7_2);
                __m256i w0 = _mm256_unpacklo_epi32(t0, t4);
                __m256i w1 = _mm256_unpackhi_epi32(t0, t4);
                __m256i w2 = _mm256_unpacklo_epi32(t1, t5);
                __m256i w3 = _mm256_unpackhi_epi32(t1, t5);
                __m256i w4 = _mm256_unpacklo_epi32(t2, t6);
                __m256i w5 = _mm256_unpackhi_epi32(t2, t6);
                __m256i w6 = _mm256_unpacklo_epi32(t3, t7);
                __m256i w7 = _mm256_unpackhi_epi32(t3, t7);
                __m256i x0 = _mm256_unpacklo_epi32(w0, w4);
                __m256i x1 = _mm256_unpackhi_epi32(w0, w4);
                __m256i x2 = _mm256_unpacklo_epi32(w1, w5);
                __m256i x3 = _mm256_unpackhi_epi32(w1, w5);
                __m256i x4 = _mm256_unpacklo_epi32(w2, w6);
                __m256i x5 = _mm256_unpackhi_epi32(w2, w6);
                __m256i x6 = _mm256_unpacklo_epi32(w3, w7);
                __m256i x7 = _mm256_unpackhi_epi32(w3, w7);

                // 4.store
                _mm256_storeu_si256((__m256i*)p, x0);
                _mm256_storeu_si256((__m256i*)(p + 8), x1);
                _mm256_storeu_si256((__m256i*)(p + 16), x2);
                _mm256_storeu_si256((__m256i*)(p + 24), x3);
                _mm256_storeu_si256((__m256i*)(p + 32), x4);
                _mm256_storeu_si256((__m256i*)(p + 40), x5);
                _mm256_storeu_si256((__m256i*)(p + 48), x6);
                _mm256_storeu_si256((__m256i*)(p + 56), x7);
        }

        // stride = 128
        for (int* p = arr; p < arr + 8192; p += 128) {
                int num = ((p - arr) >> 7);
                int desc = 0;
                while (num) {
                        desc ^= (num & 1);
                        num >>= 1;
                }
                if (!desc) {
                        bitonic_sort<128>(p);
                } else {
                        bitonic_sort_desc<128>(p);
                }
        }

        // stride = 256
        for (int* p = arr; p < arr + 8192; p += 256) {
                int num = ((p - arr) >> 8);
                int desc = 0;
                while (num) {
                        desc ^= (num & 1);
                        num >>= 1;
                }
                if (!desc) {
                        bitonic_sort<256>(p);
                } else {
                        bitonic_sort_desc<256>(p);
                }
        }

        // stride = 512
        for (int* p = arr; p < arr + 8192; p += 512) {
                int num = ((p - arr) >> 9);
                int desc = 0;
                while (num) {
                        desc ^= (num & 1);
                        num >>= 1;
                }
                if (!desc) {
                        bitonic_sort<512>(p);
                } else {
                        bitonic_sort_desc<512>(p);
                }
        }

        // stride = 1024
        for (int* p = arr; p < arr + 8192; p += 1024) {
                int num = ((p - arr) >> 10);
                int desc = 0;
                while (num) {
                        desc ^= (num & 1);
                        num >>= 1;
                }
                if (!desc) {
                        bitonic_sort<1024>(p);
                } else {
                        bitonic_sort_desc<1024>(p);
                }
        }

        // stride = 2048
        for (int* p = arr; p < arr + 8192; p += 2048) {
                int num = ((p - arr) >> 11);
                int desc = 0;
                while (num) {
                        desc ^= (num & 1);
                        num >>= 1;
                }
                if (!desc) {
                        bitonic_sort<2048>(p);
                } else {
                        bitonic_sort_desc<2048>(p);
                }
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
        make_bitonic8192(arr);
        bitonic_sort<8192>(arr);
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
