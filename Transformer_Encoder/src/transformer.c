#include "transformer.h"

// Single transformer encoder layer forward pass

Matrix_t transformer_encoder_layer_forward(const TransformerLayer_t *layer, Matrix_t input, int num_heads) {
    //Self-attention
    Matrix_t Q = matmul(&input, &layer->W_Q);
    Matrix_t K = matmul(&input, &layer->W_K);
    Matrix_t V = matmul(&input, &layer->W_V);

    Matrix_t attention_output = multi_head_attention_forward(&Q, &K, &V, &layer->W_O, num_heads);

    // Add & Norm
    Matrix_t residual_1 = init_matrix(attention_output.rows, attention_output.cols);
    add_matrices(&input, &attention_output, &residual_1);
    Matrix_t norm_1 = init_matrix(residual_1.rows, residual_1.cols);
    layer_norm(&residual_1, &layer->gamma1, &layer->beta1, &norm_1);

    Matrix_t ff_output = feed_forward(&norm_1, &layer->W1, &layer->b1, &layer->W2, &layer->b2);

    // Add & Norm
    Matrix_t residual_2 = init_matrix(ff_output.rows, ff_output.cols);
    add_matrices(&norm_1, &ff_output, &residual_2);
    Matrix_t norm_2 = init_matrix(residual_2.rows, residual_2.cols);
    layer_norm(&residual_2, &layer->gamma2, &layer->beta2, &norm_2);

    //Free intermediate matrices
    free_matrix(Q);
    free_matrix(K);
    free_matrix(V);
    free_matrix(attention_output);
    free_matrix(residual_1);
    free_matrix(norm_1);
    free_matrix(ff_output);
    free_matrix(residual_2);
} 

//Stack multiple transformer encoder layers
Matrix_t transformer_encoder_forward(const TransformerLayer_t *layers, int num_layers, Matrix_t input, int num_heads) {
    Matrix_t output = input;
    for (int i = 0; i < num_layers; i++) {
        output = transformer_encoder_layer_forward(&layers[i], output, num_heads);
    }
    return output;
}