#include <chrono>
#include <cstdlib>
#include <iostream>

#include <emmintrin.h>
#include <immintrin.h>

#include "check_sorted.h"

using namespace std;

constexpr int size = (1 << 13);
// constexpr int size = (1 << 4);
constexpr int tries = 1000;

// Lane permutations for a 16-int register: lane i ^ j, a full reversal, and
// a reversal of the upper eight lanes.
alignas(64) static const int kXor8[16] = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};
alignas(64) static const int kXor4[16] = {4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
alignas(64) static const int kXor2[16] = {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13};
alignas(64) static const int kXor1[16] = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
alignas(64) static const int kReverse[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
alignas(64) static const int kReverseUpperEight[16] = {0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8};

// One compare-exchange step on the 16 ints of a register: lane i is paired with
// lane idx[i] and keeps the larger value where bit i of max_mask is set.
inline __m512i cmpx16(__m512i a, __m512i idx, __mmask16 max_mask) {
        __m512i b = _mm512_permutexvar_epi32(idx, a);
        return _mm512_mask_blend_epi32(max_mask, _mm512_min_epi32(a, b), _mm512_max_epi32(a, b));
}

// Strides 8, 4, 2 and 1 of a bitonic merge, inside each group of sixteen.
template <bool desc>
inline void merge_within16(int* arr, int n) {
        const __m512i x8 = _mm512_load_si512((const __m512i*)kXor8);
        const __m512i x4 = _mm512_load_si512((const __m512i*)kXor4);
        const __m512i x2 = _mm512_load_si512((const __m512i*)kXor2);
        const __m512i x1 = _mm512_load_si512((const __m512i*)kXor1);
        const __mmask16 flip = desc ? 0xFFFF : 0;
        for (int* p = arr; p < arr + n; p += 16) {
                __m512i a = _mm512_load_si512((__m512i*)p);
                a = cmpx16(a, x8, 0xFF00 ^ flip);
                a = cmpx16(a, x4, 0xF0F0 ^ flip);
                a = cmpx16(a, x2, 0xCCCC ^ flip);
                a = cmpx16(a, x1, 0xAAAA ^ flip);
                _mm512_store_si512((__m512i*)p, a);
        }
}

// Compare-exchange of two 8-int registers, lane by lane: a gets the minimum.
inline void cmpx8(__m256i& a, __m256i& b) {
        __m256i lo = _mm256_min_epi32(a, b);
        b = _mm256_max_epi32(a, b);
        a = lo;
}


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
                for (int* p = arr, *q = arr + 2048, *sect_end = arr + 2048; q < arr + n; sect_end += 2 * 2048, p += 2048, q += 2048) {
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
                for (int* p = arr, *q = arr + 1024, *sect_end = arr + 1024; q < arr + n; sect_end += 2 * 1024, p += 1024, q += 1024) {
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
                for (int* p = arr, *q = arr + 512, *sect_end = arr + 512; q < arr + n; sect_end += 2 * 512, p += 512, q += 512) {
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
                for (int* p = arr, *q = arr + 256, *sect_end = arr + 256; q < arr + n; sect_end += 2 * 256, p += 256, q += 256) {
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
                for (int* p = arr, *q = arr + 128, *sect_end = arr + 128; q < arr + n; sect_end += 2 * 128, p += 128, q += 128) {
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

        merge_within16<false>(arr, n);
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
                for (int* p = arr, *q = arr + 2048, *sect_end = arr + 2048; q < arr + n; sect_end += 2 * 2048, p += 2048, q += 2048) {
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
                for (int* p = arr, *q = arr + 1024, *sect_end = arr + 1024; q < arr + n; sect_end += 2 * 1024, p += 1024, q += 1024) {
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
                for (int* p = arr, *q = arr + 512, *sect_end = arr + 512; q < arr + n; sect_end += 2 * 512, p += 512, q += 512) {
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
                for (int* p = arr, *q = arr + 256, *sect_end = arr + 256; q < arr + n; sect_end += 2 * 256, p += 256, q += 256) {
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
                for (int* p = arr, *q = arr + 128, *sect_end = arr + 128; q < arr + n; sect_end += 2 * 128, p += 128, q += 128) {
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

        merge_within16<true>(arr, n);
}

inline void make_bitonic8192(int* arr) {
        const __m512i x8 = _mm512_load_si512((const __m512i*)kXor8);
        const __m512i x4 = _mm512_load_si512((const __m512i*)kXor4);
        const __m512i x2 = _mm512_load_si512((const __m512i*)kXor2);
        const __m512i x1 = _mm512_load_si512((const __m512i*)kXor1);
        const __m512i reverse = _mm512_load_si512((const __m512i*)kReverse);
        const __m512i reverse_upper_eight = _mm512_load_si512((const __m512i*)kReverseUpperEight);

        // stride = 64,每个块内排序
        for (int* p = arr; p < arr + 8192; p += 64) {
                // 8*8 in-register sort.
                // 1.load
                __m256i row0 = _mm256_loadu_si256((__m256i*)p);
                __m256i row1 = _mm256_loadu_si256((__m256i*)(p + 8));
                __m256i row2 = _mm256_loadu_si256((__m256i*)(p + 16));
                __m256i row3 = _mm256_loadu_si256((__m256i*)(p + 24));
                __m256i row4 = _mm256_loadu_si256((__m256i*)(p + 32));
                __m256i row5 = _mm256_loadu_si256((__m256i*)(p + 40));
                __m256i row6 = _mm256_loadu_si256((__m256i*)(p + 48));
                __m256i row7 = _mm256_loadu_si256((__m256i*)(p + 56));

                // 2. Sort each lane across the eight registers with Batcher's
                // odd-even merge network: 19 compare-exchanges in 6 layers.
                cmpx8(row0, row1); cmpx8(row2, row3); cmpx8(row4, row5); cmpx8(row6, row7);
                cmpx8(row0, row2); cmpx8(row1, row3); cmpx8(row4, row6); cmpx8(row5, row7);
                cmpx8(row1, row2); cmpx8(row5, row6);
                cmpx8(row0, row4); cmpx8(row1, row5); cmpx8(row2, row6); cmpx8(row3, row7);
                cmpx8(row2, row4); cmpx8(row3, row5);
                cmpx8(row1, row2); cmpx8(row3, row4); cmpx8(row5, row6);

                // 3. 8x8 transpose: afterwards each register holds one sorted lane.
                __m256i t0 = _mm256_unpacklo_epi32(row0, row1);
                __m256i t1 = _mm256_unpackhi_epi32(row0, row1);
                __m256i t2 = _mm256_unpacklo_epi32(row2, row3);
                __m256i t3 = _mm256_unpackhi_epi32(row2, row3);
                __m256i t4 = _mm256_unpacklo_epi32(row4, row5);
                __m256i t5 = _mm256_unpackhi_epi32(row4, row5);
                __m256i t6 = _mm256_unpacklo_epi32(row6, row7);
                __m256i t7 = _mm256_unpackhi_epi32(row6, row7);
                __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
                __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
                __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
                __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
                __m256i u4 = _mm256_unpacklo_epi64(t4, t6);
                __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
                __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
                __m256i u7 = _mm256_unpackhi_epi64(t5, t7);
                const __m256i runs[8] = {
                        _mm256_permute2x128_si256(u0, u4, 0x20), _mm256_permute2x128_si256(u1, u5, 0x20),
                        _mm256_permute2x128_si256(u2, u6, 0x20), _mm256_permute2x128_si256(u3, u7, 0x20),
                        _mm256_permute2x128_si256(u0, u4, 0x31), _mm256_permute2x128_si256(u1, u5, 0x31),
                        _mm256_permute2x128_si256(u2, u6, 0x31), _mm256_permute2x128_si256(u3, u7, 0x31),
                };

                // 4. Merge pairs of runs of eight into runs of sixteen: reverse the
                // second run so the pair is bitonic, merge ascending, and reverse
                // the blocks that the next merge needs in descending order (same
                // parity rule as the merges below).
                for (int k = 0; k < 4; ++k) {
                        int* q = p + 16 * k;
                        __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(runs[2 * k]), runs[2 * k + 1], 1);
                        v = _mm512_permutexvar_epi32(reverse_upper_eight, v);
                        v = cmpx16(v, x8, 0xFF00);
                        v = cmpx16(v, x4, 0xF0F0);
                        v = cmpx16(v, x2, 0xCCCC);
                        v = cmpx16(v, x1, 0xAAAA);
                        if (__builtin_parity((unsigned)((q - arr) >> 4))) {
                                v = _mm512_permutexvar_epi32(reverse, v);
                        }
                        _mm512_store_si512((__m512i*)q, v);
                }
        }

        // stride = 32 and 64
        for (int* p = arr; p < arr + 8192; p += 32) {
                if (__builtin_parity((unsigned)((p - arr) >> 5))) {
                        bitonic_sort_desc<32>(p);
                } else {
                        bitonic_sort<32>(p);
                }
        }
        for (int* p = arr; p < arr + 8192; p += 64) {
                if (__builtin_parity((unsigned)((p - arr) >> 6))) {
                        bitonic_sort_desc<64>(p);
                } else {
                        bitonic_sort<64>(p);
                }
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

        // stride = 4096: ascending then descending, so the final merge sees a
        // bitonic sequence.
        bitonic_sort<4096>(arr);
        bitonic_sort_desc<4096>(arr + 4096);
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
        sort_check.expect(arr, size);

        auto start = chrono::high_resolution_clock::now();
        make_bitonic8192(arr);
        bitonic_sort<8192>(arr);
        auto end = chrono::high_resolution_clock::now();
        sort_check.verify(arr, size);
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
        return sort_check.report();
}
