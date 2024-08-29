#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace std;

constexpr int size = (1 << 13);
constexpr int tries = 100;

inline void compareAndSwap(int* p, int* q, bool ascend) {
        if ((ascend && *p > *q) || (!ascend && *p < *q)) {
                int tmp = *p;
                *p = *q;
                *q = tmp;
        }
}

void bitonic_sort(int* arr, int n, bool ascend) {
        if (n == 1) {
                return;
        }

        bitonic_sort(arr, (n >> 1), ascend);
        bitonic_sort(arr + (n >> 1), n - (n >> 1), !ascend);
        for (int* p = arr, *q = arr + (n >> 1); p < arr + (n >> 1); ++p, ++q) {
                compareAndSwap(p, q, ascend);
        }
        bitonic_sort(arr, (n >> 1), ascend);
        bitonic_sort(arr + (n >> 1), n - (n >> 1), ascend);
}

double test(int* arr) {
        for (int i = 0; i < size; ++i) {
                arr[i] = (rand() % 0x3ff);
        }

        auto start = chrono::high_resolution_clock::now();
        bitonic_sort(arr, size, true);
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
        // cout << endl;
        cout << "n=" << size << ", average time: " << millis / tries << " ms." << endl;
        delete[] arr;
        return 0;
}
