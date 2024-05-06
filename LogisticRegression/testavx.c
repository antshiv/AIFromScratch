#include <immintrin.h>
#include <stdio.h>
#include <time.h>

int main() {
    clock_t t = clock();
    __m256 a = _mm256_setr_ps(1,2,3,4,5,6,7,8);
    __m256 b = _mm256_setr_ps(1,2,3,4,5,6,7,8);
    __m256 c = _mm256_mul_ps(a, b);
    float *result = (float*)&c;
    for (int i = 0; i < 8; i++) {
        printf("%f ", result[i]);
    }
    t = clock() - t;
    printf("\nTime taken %f\n", ((double)t)/CLOCKS_PER_SEC);

    float scalar[] = {1,2,3,4,5,6,7,8};
    for (int i = 0; i < 8; i++) {
        scalar[i] = scalar[i] * scalar[i];
    }
    for (int i = 0; i < 8; i++) {
        printf("%f ", scalar[i]);
    }
    t = clock() - t;
    printf("\nTime taken %f\n", ((double)t)/CLOCKS_PER_SEC);

    return 0;
}