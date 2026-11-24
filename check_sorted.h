#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

// Compares every timed sort with std::sort, element by element.
// expect() runs before the timer starts and verify() after it stops, so the
// check is not included in the reported time.
struct SortCheck {
        std::vector<int> expected;
        int checked = 0;
        int failed = 0;

        void expect(const int* input, int n) {
                expected.assign(input, input + n);
                std::sort(expected.begin(), expected.end());
        }

        void verify(const int* output, int n) {
                ++checked;
                if (!std::equal(expected.begin(), expected.end(), output)) {
                        ++failed;
                }
        }

        int report() const {
                if (failed) {
                        printf("CHECK FAILED: %d of %d outputs are not correctly sorted.\n", failed, checked);
                        return 1;
                }
                printf("CHECK PASSED: all %d outputs are correctly sorted.\n", checked);
                return 0;
        }
};

static SortCheck sort_check;
