#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace std;

constexpr int size = (1 << 13);
constexpr int tries = 1000;

int asc_compare(const void* a, const void* b) {
        int arg1 = *(const int *)a;
        int arg2 = *(const int *)b;
        return arg1 - arg2;
}

double test(int* arr) {
        for (int i = 0; i < size; ++i) {
                arr[i] = (rand() % 0x3ff);
        }

        auto start = chrono::high_resolution_clock::now();
        qsort(arr, size, sizeof(int), asc_compare);
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
