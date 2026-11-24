# cpu_bitonicsort

在 CPU 上手写双调排序（bitonic sort）：从朴素实现开始，逐步改成非递归、SSE、AVX2、AVX-512 和寄存器内排序（in-register sort）。输入固定为 n = 8192 个 `int`，取值 0–1022。

> **更正（2026-11）**：本仓库早期 README 和 2024 年博客中“6 倍于 std::sort”的结论已经撤回。原代码从不检查排序结果；补上逐元素校验后发现，从第 2 版开始的各个版本都没有把数组排好序，后面几版之所以快，是因为大部分比较根本没有执行。

## 发现并修复的错误

1. **合并循环每层只处理了第一个分块**（第 2–8 版，共 44 处）：`sect_end` 每次只前进一个 `stride`，和 `p` 始终相等，从第二个分块开始内层循环一次都不执行。已改为 `sect_end += 2 * stride`。
2. **第 3 版漏掉了最后两层比较**：只比较到步长 4，步长 2 和 1 没有做，每 4 个元素内部是乱序的。已用 `_mm_shuffle_epi32` 加 `_mm_blend_epi16` 在寄存器内补上。

## 当前状态

每个程序都会把每次排序的结果与 `std::sort` 逐元素比较（`check_sorted.h`，比较不计入耗时），最后输出 `CHECK PASSED` 或 `CHECK FAILED`；校验失败时退出码非 0。

| 版本 | 内容 | 结果校验 |
|---|---|---|
| `0_base.cc` | 朴素递归实现，O(n²) | ✅ 通过 |
| `1_nlogn.cc` | 两次 `qsort` 构造双调序列，再递归双调合并，O(n log n) | ✅ 通过 |
| `2_norecursion.cc` | 合并改为非递归 | ✅ 通过（已修复） |
| `3_simd.cc` | 最后几层用 SSE | ✅ 通过（已修复） |
| `4_specialized.cc` | 针对 n = 8192 展开 | ❌ 仍有错误 |
| `5_simd8.cc` | AVX2（256 位） | ❌ 仍有错误 |
| `6_simd16.cc` | AVX-512 | ❌ 仍有错误 |
| `7_noqsort.cc` | 自己构造双调序列，不再调用 `qsort` | ❌ 仍有错误 |
| `8_inregistersort8x8.cc` | 寄存器内 8×8 排序 | ❌ 仍有错误 |
| `libc_qsort.cc`、`stl_sort.cc` | 基线 | ✅ 通过 |

第 4–8 版修好之前，它们的耗时没有意义。

## 性能

在我能测试的环境里，通过校验的版本都**比 `std::sort` 慢**：

- Compiler Explorer 的 x86 执行机（Intel Xeon E5-2686 v4，支持 AVX2，共享云主机，同一版本几轮之间耗时相差 2–3 倍）；
- Apple Silicon，x86 SIMD 指令通过 [SIMDe](https://github.com/simd-everywhere/simde) 翻译。

第 1–3 版的大部分时间花在构造双调序列的两次 `qsort` 上。因为共享主机的波动太大，这里不列具体数字。

## 运行

```bash
./run_3_simd.sh          # 编译并运行指定版本，输出平均耗时和校验结果
./run_stl_sort.sh        # std::sort 基线
./run_libc_qsort.sh      # qsort 基线
```

第 3–5 版需要支持 AVX2 的 x86 CPU，第 6–8 版需要支持 AVX-512 的 x86 CPU。

---

Hand-written bitonic sort on the CPU, from a naive version through a non-recursive merge, SSE, AVX2, AVX-512 and in-register sorting. The input is fixed at n = 8192 `int`s in the range 0–1022.

**Correction (November 2026):** the earlier claim, in this README and in a 2024 blog post, that this code sorts 6× faster than `std::sort` is retracted. The original programs never checked their output. With an element-wise check against `std::sort`, every version from 2 onward failed to sort; the later versions were fast because most of the comparisons never ran.

Two bugs are now fixed: the merge loop advanced `sect_end` by one stride instead of two, so each level processed only its first block (44 places in versions 2–8); and version 3 stopped at stride 4, skipping strides 2 and 1. Versions 0–3 and both baselines now pass the check; versions 4–8 still fail and their timings are meaningless until they are fixed. In every environment I could test (a shared AVX2 cloud VM on Compiler Explorer, and Apple Silicon via SIMDe), every version that passes the check is slower than `std::sort`.

Each program checks every sort against `std::sort` (`check_sorted.h`, outside the timed region), prints `CHECK PASSED` or `CHECK FAILED`, and exits non-zero on failure.
