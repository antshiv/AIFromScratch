// transformer_backward.c

#include "transformer_backward.h"
#include "attention_backward.h"
#include "feedforward_backward.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void transformer_encoder_backward(TransformerLayer_t *layers, int num_layers, Matrix_t d_output, Matrix_t input, int num_heads, Matrix_t *d_input) {
    Matrix_t d_intermediate = d_output; // Initial gradient

    // Backward pass through each layer
    for (int i = num_layers - 1; i >= 0; i--) {
        // Backward pass through feedforward layer
        feedforward_backward(d_intermediate, layers[i].feedforward_input, layers[i].W1, layers[i].b1, layers[i].W2, layers[i].b2, &layers[i].d_W1, &layers[i].d_b1, &layers[i].d_W2, &layers[i].d_b2, &d_intermediate);

        // Backward pass through attention layer
        multi_head_attention_backward(d_intermediate, layers[i].Q, layers[i].K, layers[i].V, layers[i].W_Q, layers[i].W_K, layers[i].W_V, layers[i].W_O, num_heads, &layers[i].d_Q, &layers[i].d_K, &layers[i].d_V, &layers[i].d_W_Q, &layers[i].d_W_K, &layers[i].d_W_V, &layers[i].d_W_O);

        // Backward pass through layer normalization and residual connection
    }

    // Set final gradient for input
    *d_input = d_intermediate;
}
