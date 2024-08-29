#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

constexpr int size = (1 << 13);
constexpr int tries = 1000;

double test(vector<int>& arr) {
        for (int i = 0; i < size; ++i) {
                arr[i] = (rand() % 0x3ff);
        }

        auto start = chrono::high_resolution_clock::now();
        sort(arr.begin(), arr.end());
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        return elapsed.count() * 1000;
}

int main() {
        srand(100);
        vector<int> arr(size);
        double millis = 0.0;
        for (int i = 0; i < tries; ++i) {
                millis += test(arr);
        }
        // for (int i = 0; i < size; ++i) {
        //      cout << arr[i] << ", ";
        // }
        // cout << endl;
        cout << "n=" << size << ", average time: " << millis / tries << " ms." << endl;
        return 0;
}
