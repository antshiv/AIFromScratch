#include <stdlib.h>
#include <stdio.h>
#include <stdalign.h> // For alignment macros
#include "transformer.h"

Matrix_t create_matrix(int rows, int cols, int alignment) {
    Matrix_t matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.stride = cols; // Assuming no padding
    matrix.alignment = alignment;

    // Allocate aligned memory
    if (posix_memalign((void**)&matrix.data, alignment, rows * cols * sizeof(float)) != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

void free_matrix(Matrix_t matrix) {
    free(matrix.data);
    matrix.data = NULL;
}

void print_matrix(Matrix_t matrix) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            printf("%f ", matrix.data[i * matrix.stride + j]);
        }
        printf("\n");
    }
}

void xavier_init(Matrix_t *matrix) {
    // Initialize the matrix with Xavier initialization
    float scale = sqrtf(2.0f / (matrix->rows + matrix->cols));
    for (int i = 0; i < matrix->rows * matrix->stride; i++) {
        matrix->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

/* Calculation the Attention heads and all the utiltiies needed */

void position_encoding(Matrix_t *matrix) {
    // Implement position encoding
}

void attention_heads(TransformerLayer_t *layer, Matrix_t *input) {
    // Implement attention heads calculation
}

void linear_transformer(TransformerLayer_t *layer, Matrix_t *input) {
    // Implement linear transformer calculation
}

void layer_norm_residual(TransformerLayer_t *layer, Matrix_t *input, Matrix_t *output) {
    // Implement layer normalization and residual connection
}

void ner_classification_layer(Transformer_t *transformer, Matrix_t *input) {
    // Implement NER classification layer
}

void backpropagation(Transformer_t *transformer, Matrix_t *input, Matrix_t *output) {
    // Implement backpropagation
}

void calculate_loss(Matrix_t *predictions, Matrix_t *labels) {
    // Implement loss calculation (cross-entropy)
}

void adam_optimizer_update(Transformer_t *transformer) {
    // Implement Adam optimizer update
}
