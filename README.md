# cpu_bitonicsort

在 CPU 上手写双调排序（bitonic sort）：从朴素实现开始，逐步改成非递归、SSE、AVX2、AVX-512 和寄存器内排序（in-register sort）。输入固定为 n = 8192 个 `int`，取值 0–1022。

> **更正（2026-11）**：本仓库早期 README 和 2024 年博客中“6 倍于 std::sort”的结论已经撤回。原代码从不检查排序结果；补上逐元素校验后发现，从第 2 版开始的各个版本都没有把数组排好序，后面几版之所以快，是因为大部分比较根本没有执行。修复后的结果见下文“性能”一节，那是在另一台机器上重新测得的，和 2024 年的数字不可比。

## 发现并修复的错误

1. **合并循环每层只处理了第一个分块**（第 2–8 版，共 44 处）：`sect_end` 每次只前进一个 `stride`，和 `p` 始终相等，从第二个分块开始内层循环一次都不执行。已改为 `sect_end += 2 * stride`。
2. **第 3、4 版漏掉了最后两层比较**：只比较到步长 4，步长 2 和 1 没有做，每 4 个元素内部是乱序的。已用 `_mm_shuffle_epi32` 加 `_mm_blend_epi16` 在寄存器内补上。
3. **第 5 版另有两处错误**：步长 16 这一层用的是 128 位寄存器，指针却每次前进 8 个数，每 8 个数只比较了前 4 个；而且只做到步长 8，漏掉了步长 4、2、1 三层。已改用 256 位寄存器，并在寄存器内补上最后三层。
4. **第 6 版漏掉了最后四层比较**：只做到步长 16，步长 8、4、2、1 没有做，每 16 个元素内部是乱序的。已在 512 位寄存器内补上（`_mm512_shuffle_i64x2`、`_mm512_shuffle_epi32` 加掩码 blend）。
5. **第 7 版有三处缺失**：
   - 升序、降序两个合并函数都只做到步长 16，漏掉了步长 8、4、2、1；
   - 4×4 寄存器内排序只把每 16 个数排成 4 段有序的 4 个数，缺少 4→8→16 两层合并，下一层合并 32 个数时拿到的不是双调序列；
   - 构造双调序列时只合并到 2048，漏掉了合并到 4096 的一层，最后一次合并 8192 个数时，输入是 4 段 2048 而不是双调序列。

   已全部补上：寄存器内的比较交换用 `_mm512_permutexvar_epi32` 加掩码 blend；4×4 排序后在 512 位寄存器内反转第 2、4 段，再完成 8 和 16 两层合并；最后补上 4096 这一层（前一半升序、后一半降序）。
6. **第 8 版的 8×8 寄存器内排序本身是错的**：
   - 排序网络只有 3 轮共 12 次比较，而排好 8 个数至少需要 19 次比较、6 层，所以每一列都没有排好；
   - 8×8 转置只做了三次 32 位交错，缺少 64 位交错和高低 128 位交换；
   - 另外有和第 7 版相同的三处缺失：合并只做到步长 16，缺少从 8 个一段到 64 个一段的合并，缺少 4096 这一层。

   已改用 Batcher 奇偶归并网络（19 次比较、6 层）和标准的 8×8 转置（`_mm256_unpacklo/hi_epi32`、`_mm256_unpacklo/hi_epi64`、`_mm256_permute2x128_si256`），再在 512 位寄存器内把 8 个一段合并成 16 个一段，之后用合并函数合并到 32 和 64，其余与第 7 版相同。

## 当前状态

每个程序都会把每次排序的结果与 `std::sort` 逐元素比较（`check_sorted.h`，比较不计入耗时），最后输出 `CHECK PASSED` 或 `CHECK FAILED`；校验失败时退出码非 0。

| 版本 | 内容 | 结果校验 |
|---|---|---|
| `0_base.cc` | 朴素递归实现，O(n²) | ✅ 通过 |
| `1_nlogn.cc` | 两次 `qsort` 构造双调序列，再递归双调合并，O(n log n) | ✅ 通过 |
| `2_norecursion.cc` | 合并改为非递归 | ✅ 通过（已修复） |
| `3_simd.cc` | 最后几层用 SSE | ✅ 通过（已修复） |
| `4_specialized.cc` | 针对 n = 8192 展开 | ✅ 通过（已修复） |
| `5_simd8.cc` | AVX2（256 位） | ✅ 通过（已修复） |
| `6_simd16.cc` | AVX-512 | ✅ 通过（已修复） |
| `7_noqsort.cc` | 自己构造双调序列，不再调用 `qsort` | ✅ 通过（已修复） |
| `8_inregistersort8x8.cc` | 寄存器内 8×8 排序 | ✅ 通过（已修复） |
| `libc_qsort.cc`、`stl_sort.cc` | 基线 | ✅ 通过 |

第 7、8 版另外用其他几种输入测试过，全部正确：全范围随机 `int`（含负数）、已排序、逆序、全部相同、随机 0/1、`INT_MIN` 与 `INT_MAX` 混合。

## 性能

阿里云 ECS，Intel Xeon 6982P-C（4 vCPU，支持 AVX-512），GCC 15.2，`-O3`。每个版本运行 3 次，每次取 1000 次排序的平均值（第 0 版是 100 次），表中是 3 次的中位数，3 次之间相差不到 1%。

| 版本 | 每次排序耗时 | 相对 `std::sort` 的速度 |
|---|---|---|
| `0_base` | 81.3 ms | 0.004× |
| `1_nlogn` | 0.574 ms | 0.53× |
| `2_norecursion` | 0.557 ms | 0.55× |
| `3_simd`（SSE） | 0.511 ms | 0.59× |
| `4_specialized` | 0.509 ms | 0.60× |
| `5_simd8`（AVX2） | 0.505 ms | 0.60× |
| `6_simd16`（AVX-512） | 0.502 ms | 0.60× |
| **`7_noqsort`** | **0.0211 ms** | **14.4×** |
| **`8_inregistersort8x8`** | **0.0200 ms** | **15.2×** |
| `libc_qsort` | 0.523 ms | 0.58× |
| `std::sort` | 0.304 ms | 1× |

第 1–6 版用两次 `qsort` 构造双调序列，这一步约 0.50 ms，占了总耗时的大部分。SIMD 只加速了后面的双调合并：单独计时，合并部分从标量的 35.8 µs（第 2 版）降到 SSE 的 11.5 µs（第 3 版）、AVX2 的 4.5 µs（第 5 版）和 AVX-512 的 3.1 µs（第 6 版）。第 7 版改用寄存器内排序加 SIMD 合并来构造双调序列，整个排序约 21 µs，比 `std::sort` 快约 14 倍；第 8 版把寄存器内排序从 4×4 换成 8×8，约 20 µs，快约 15 倍。

这些数字只适用于这台机器、这个固定规模（8192 个 `int`，32 KB，能放进 L1 缓存）和这种输入。

## 运行

```bash
./run_7_noqsort.sh       # 编译并运行指定版本，输出平均耗时和校验结果
./run_stl_sort.sh        # std::sort 基线
./run_libc_qsort.sh      # qsort 基线
```

第 3–5 版需要支持 AVX2 的 x86 CPU，第 6–8 版需要支持 AVX-512 的 x86 CPU。

---

Hand-written bitonic sort on the CPU, from a naive version through a non-recursive merge, SSE, AVX2, AVX-512 and in-register sorting. The input is fixed at n = 8192 `int`s in the range 0–1022.

**Correction (November 2026):** the earlier claim, in this README and in a 2024 blog post, that this code sorts 6× faster than `std::sort` is retracted. The original programs never checked their output. With an element-wise check against `std::sort`, every version from 2 onward failed to sort; the later versions were fast because most of the comparisons never ran. The results below were measured after the fixes, on a different machine, and are not comparable with the 2024 numbers.

These bugs are now fixed: the merge loop advanced `sect_end` by one stride instead of two, so each level processed only its first block (44 places in versions 2–8); versions 3 and 4 stopped at stride 4, skipping strides 2 and 1; version 5 compared only half of each 8-element chunk at stride 16 and stopped at stride 8; version 6 stopped at stride 16; version 7 stopped its merges at stride 16, never merged the 4×4 in-register runs into runs of 16, and skipped the merge to 4096 before the final merge; and version 8's 8×8 in-register sort used an incomplete sorting network (12 comparators where 19 are needed) and an incorrect transpose, on top of the same three gaps as version 7. All versions and both baselines now pass the check. Versions 7 and 8 were also checked on full-range random ints, sorted, reverse-sorted, all-equal, random 0/1 and `INT_MIN`/`INT_MAX` inputs.

On an Intel Xeon 6982P-C (Alibaba Cloud ECS, AVX-512, GCC 15.2, `-O3`), version 7 sorts 8192 ints in 0.0211 ms and version 8 in 0.0200 ms, versus 0.304 ms for `std::sort`: about 14× and 15× faster. Versions 1–6 are all slower than `std::sort`: their two `qsort` calls take about 0.50 ms, and SIMD speeds up only the merge (35.8 µs scalar, 3.1 µs AVX-512). These numbers apply to this machine, this fixed size (32 KB, which fits in L1) and this input.

Each program checks every sort against `std::sort` (`check_sorted.h`, outside the timed region), prints `CHECK PASSED` or `CHECK FAILED`, and exits non-zero on failure.
