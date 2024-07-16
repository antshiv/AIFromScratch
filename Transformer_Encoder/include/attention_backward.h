// attention_backward.h

#ifndef ATTENTION_BACKWARD_H
#define ATTENTION_BACKWARD_H

#include "matrix.h"

// Function prototypes
void multi_head_attention_backward(Matrix_t d_output, Matrix_t Q, Matrix_t K, Matrix_t V, Matrix_t W_Q, Matrix_t W_K, Matrix_t W_V, Matrix_t W_O, int num_heads, Matrix_t *d_Q, Matrix_t *d_K, Matrix_t *d_V, Matrix_t *d_W_Q, Matrix_t *d_W_K, Matrix_t *d_W_V, Matrix_t *d_W_O);

#endif // ATTENTION_BACKWARD_H
