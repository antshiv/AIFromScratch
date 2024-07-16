// transformer_backward.h

#ifndef TRANSFORMER_BACKWARD_H
#define TRANSFORMER_BACKWARD_H

#include "transformer.h"

// Function prototypes
void transformer_encoder_backward(TransformerLayer_t *layers, int num_layers, Matrix_t d_output, Matrix_t input, int num_heads, Matrix_t *d_input);

#endif // TRANSFORMER_BACKWARD_H
