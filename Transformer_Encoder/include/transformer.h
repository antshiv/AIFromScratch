#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "matrix.h"
#include "feedforward.h"
#include "attention.h"
//#include "layer_norm.h"
//#include "utils.h"

typedef struct TransformerLayer {
    Matrix_t W_Q, W_K, W_V, W_O;
    Matrix_t gamma1, beta1, gamma2, beta2;
    Matrix_t W1, b1, W2, b2;
} TransformerLayer_t;

Matrix_t transformer_encoder_layer_forward(const TransformerLayer_t *layer, Matrix_t input, int num_heads);
Matrix_t transformer_encoder_forward(const TransformerLayer_t *layers, int num_layers, Matrix_t input, int num_heads);

#endif // TRANSFORMER_H
