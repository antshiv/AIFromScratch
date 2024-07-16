// feedforward_backward.h

#ifndef FEEDFORWARD_BACKWARD_H
#define FEEDFORWARD_BACKWARD_H

#include "matrix.h"

// Function prototypes
void feedforward_backward(Matrix_t d_output, Matrix_t input, Matrix_t W1, Matrix_t b1, Matrix_t W2, Matrix_t b2, Matrix_t *d_W1, Matrix_t *d_b1, Matrix_t *d_W2, Matrix_t *d_b2, Matrix_t *d_input);

#endif // FEEDFORWARD_BACKWARD_H
