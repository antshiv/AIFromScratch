#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>
#include "matrix.h"


void softmax(Matrix_t *mat);
Matrix_t attention_forward(const Matrix_t *Q, const Matrix_t *K, const Matrix_t *V);
Matrix_t multi_head_attention_forward(const Matrix_t *Q, const Matrix_t *K, const Matrix_t *V, const Matrix_t *W_O, int num_heads);
void add_matrices(const Matrix_t *A, const Matrix_t *B, Matrix_t *C);
void layer_norm(const Matrix_t *X, const Matrix_t *gamma, const Matrix_t *beta, Matrix_t *Y);

#endif // ATTENTION_H
