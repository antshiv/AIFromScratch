#include "embedding.h"
#include <stdlib.h>
#include <math.h>

// Function to initialize embeddings randomly
Matrix_t init_random_embeddings(int vocab_size, int embedding_dim) {
    Matrix_t embeddings = init_matrix(vocab_size, embedding_dim);
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            embeddings.data[i * embedding_dim + j] = ((float)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
        }
    }
    return embeddings;
}

// Function to initialize embeddings using Xavier/Glorot initialization
Matrix_t init_xavier_embeddings(int vocab_size, int embedding_dim) {
    Matrix_t embeddings = init_matrix(vocab_size, embedding_dim);
    float scale = sqrt(6.0 / (vocab_size + embedding_dim));
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            embeddings.data[i * embedding_dim + j] = ((float)rand() / RAND_MAX) * 2 * scale - scale; // Random values between -scale and +scale
        }
    }
    return embeddings;
}
