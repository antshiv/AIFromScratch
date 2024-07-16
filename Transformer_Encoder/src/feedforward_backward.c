// feedforward_backward.c

#include "feedforward_backward.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void feedforward_backward(Matrix_t d_output, Matrix_t input, Matrix_t W1, Matrix_t b1, Matrix_t W2, Matrix_t b2, Matrix_t *d_W1, Matrix_t *d_b1, Matrix_t *d_W2, Matrix_t *d_b2, Matrix_t *d_input) {
    // Implement the backward pass for the feedforward network
    // Compute gradients for W1, b1, W2, b2
    // Compute gradient for input
}
