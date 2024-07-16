#include <stdio.h>
#include <stdlib.h>
#include "transformer.h"
#include "positional_encoding.h"
#include "embedding.h"

int main() {
    int seq_len = 10;
    int d_model = 512;
    int num_heads = 8;
    int num_layers = 6;
    int vocab_size = 32000;
    int embedding_dim = 512;

    // Initialize embeddings
    Matrix_t embeddings = init_random_embeddings(vocab_size, embedding_dim);
    // Alternatively, use Xavier initialization
    // Matrix embeddings = init_xavier_embeddings(vocab_size, embedding_dim);


    // Initialize input matrix (for example purposes)
    Matrix_t input = init_matrix(seq_len, d_model);
    for (int i = 0; i < input.rows * input.cols; i++) {
        input.data[i] = (float)rand() / RAND_MAX;
    }

    // Create positional encoding and add it to the input
    Matrix_t PE = positional_encoding(seq_len, d_model);
    for (int i = 0; i < input.rows * input.cols; i++) {
        input.data[i] += PE.data[i];
    }

    // Initialize transformer layers
    TransformerLayer_t layers[num_layers];
    for (int i = 0; i < num_layers; i++) {
        layers[i].W_Q = init_matrix(d_model, d_model);
        layers[i].W_K = init_matrix(d_model, d_model);
        layers[i].W_V = init_matrix(d_model, d_model);
        layers[i].W_O = init_matrix(d_model, d_model);
        layers[i].gamma1 = init_matrix(1, d_model);
        layers[i].beta1 = init_matrix(1, d_model);
        layers[i].gamma2 = init_matrix(1, d_model);
        layers[i].beta2 = init_matrix(1, d_model);
        layers[i].W1 = init_matrix(d_model, d_model * 4);
        layers[i].b1 = init_matrix(1, d_model * 4);
        layers[i].W2 = init_matrix(d_model * 4, d_model);
        layers[i].b2 = init_matrix(1, d_model);
    }

    // Forward pass through transformer encoder
    Matrix_t output = transformer_encoder_forward(layers, num_layers, input, num_heads);

    // Print the output (for example purposes)
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            printf("%f ", output.data[i * output.cols + j]);
        }
        printf("\n");
    }

    // Free matrices
    free_matrix(input);
    free_matrix(PE);
    for (int i = 0; i < num_layers; i++) {
        free_matrix(layers[i].W_Q);
        free_matrix(layers[i].W_K);
        free_matrix(layers[i].W_V);
        free_matrix(layers[i].W_O);
        free_matrix(layers[i].gamma1);
        free_matrix(layers[i].beta1);
        free_matrix(layers[i].gamma2);
        free_matrix(layers[i].beta2);
        free_matrix(layers[i].W1);
        free_matrix(layers[i].b1);
        free_matrix(layers[i].W2);
        free_matrix(layers[i].b2);
    }
    free_matrix(output);

    return 0;
}
