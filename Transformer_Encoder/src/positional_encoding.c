#include "positional_encoding.h"
#include <math.h>
#include <stdlib.h>

// Positional encoding function
Matrix_t positional_encoding(int seq_len, int d_model) {
    Matrix_t PE = init_matrix(seq_len, d_model);
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            PE.data[pos * d_model + i] = sin(pos / pow(10000, 2.0 * i / d_model));
            if (i + 1 < d_model) {
                PE.data[pos * d_model + i + 1] = cos(pos / pow(10000, 2.0 * (i + 1) / d_model));
            }
        }
    }
    return PE;
}
