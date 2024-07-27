#include <stdlib.h>
#include <stdio.h>
#include <stdalign.h> // For alignment macros
#include "transformer.h"
#include <math.h>

// Function to create a matrix with aligned memory
Matrix_t* create_matrix(int rows, int cols, int alignment) {
    Matrix_t *matrix = (Matrix_t *)malloc(sizeof(Matrix_t));
    if (!matrix) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->stride = cols; // Assuming no padding
    matrix->alignment = alignment;

    // Allocate aligned memory
    if (posix_memalign((void**)&matrix->data, alignment, rows * cols * sizeof(float)) != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }

    return matrix;
}

// Function to free a matrix
void free_matrix(Matrix_t *matrix) {
    if (matrix) {
        free(matrix->data);
        free(matrix);
    }
}

// Function to print a matrix
void print_matrix(const Matrix_t *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->data[i * matrix->stride + j]);
        }
        printf("\n");
    }
}

// Function to perform matrix multiplication
Matrix_t* matmul(const Matrix_t *A, const Matrix_t *B) {
    if (A->cols != B->rows) {
        fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication.\n");
        exit(EXIT_FAILURE);
    }

    Matrix_t *C = create_matrix(A->rows, B->cols, A->alignment);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            C->data[i * C->cols + j] = 0;
            for (int k = 0; k < A->cols; k++) {
                C->data[i * C->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }
    return C;
}

// Function to initialize a matrix with Xavier initialization
void xavier_init(Matrix_t *matrix) {
    float scale = sqrtf(2.0f / (matrix->rows + matrix->cols));
    for (int i = 0; i < matrix->rows * matrix->stride; i++) {
        matrix->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

// Function to transpose a matrix
Matrix_t* transpose(const Matrix_t *mat) {
    Matrix_t *transposed = create_matrix(mat->cols, mat->rows, mat->alignment);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            transposed->data[j * transposed->cols + i] = mat->data[i * mat->cols + j];
        }
    }
    return transposed;
}

// Function to apply softmax to a matrix
void softmax(Matrix_t *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        float max_val = matrix->data[i * matrix->stride];
        for (int j = 1; j < matrix->cols; j++) {
            if (matrix->data[i * matrix->stride + j] > max_val) {
                max_val = matrix->data[i * matrix->stride + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i * matrix->stride + j] = expf(matrix->data[i * matrix->stride + j] - max_val);
            sum += matrix->data[i * matrix->stride + j];
        }

        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i * matrix->stride + j] /= sum;
        }
    }
}

// Function to apply position encoding to a matrix
void position_encoding(Matrix_t *matrix) {
    for (int pos = 0; pos < matrix->rows; pos++) {
        for (int i = 0; i < matrix->cols; i++) {
            if (i % 2 == 0) {
                matrix->data[pos * matrix->stride + i] = sinf(pos / powf(10000.0f, (2.0f * i) / matrix->cols));
            } else {
                matrix->data[pos * matrix->stride + i] = cosf(pos / powf(10000.0f, (2.0f * (i - 1)) / matrix->cols));
            }
        }
    }
}

void relu(Matrix_t *matrix) {
    for (int i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = fmaxf(0, matrix->data[i]);
    }
}

void layer_normalization(Matrix_t *matrix, const Matrix_t *gamma, const Matrix_t *beta) {
    for (int i = 0; i < matrix->rows; i++) {
        float mean = 0.0f;
        float variance = 0.0f;

        for (int j = 0; j < matrix->cols; j++) {
            mean += matrix->data[i * matrix->cols + j];
        }
        mean /= matrix->cols;

        for (int j = 0; j < matrix->cols; j++) {
            variance += powf(matrix->data[i * matrix->cols + j] - mean, 2);
        }
        variance /= matrix->cols;

        float stddev = sqrtf(variance + 1e-5);

        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i * matrix->cols + j] = (matrix->data[i * matrix->cols + j] - mean) / stddev;
            matrix->data[i * matrix->cols + j] = matrix->data[i * matrix->cols + j] * gamma->data[j] + beta->data[j];
        }
    }
}

Matrix_t *attention_heads(Transformer_t *transformer, Matrix_t *input) {
    int num_heads = transformer->num_heads;
    TransformerLayer_t *layer = transformer->layers;
    int head_dim = input->cols / num_heads;

    Matrix_t **Q_heads = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t **K_heads = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t **V_heads = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t **attention_output_h = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t *attention_output = create_matrix(input->rows, input->cols, 64);

    for (int i = 0; i < num_heads; i++) {
        Q_heads[i] = create_matrix(input->rows, head_dim, 64);
        K_heads[i] = create_matrix(input->rows, head_dim, 64);
        V_heads[i] = create_matrix(input->rows, head_dim, 64);

        for (int j = 0; j < input->rows; j++) {
            for (int k = 0; k < head_dim; k++) {
                Q_heads[i]->data[j * head_dim + k] = input->data[j * input->cols + i * head_dim + k];
                K_heads[i]->data[j * head_dim + k] = input->data[j * input->cols + i * head_dim + k];
                V_heads[i]->data[j * head_dim + k] = input->data[j * input->cols + i * head_dim + k];
            }
        }

        // Compute attention for each head
        Matrix_t *QWQ = matmul(Q_heads[i], &layer->W_Q);
        Matrix_t *KWK = matmul(K_heads[i], &layer->W_K);
        Matrix_t *VWV = matmul(V_heads[i], &layer->W_V);

        // Scaled dot-product attention
        Matrix_t *K_T = transpose(KWK);
        Matrix_t *QK = matmul(QWQ, K_T);

        // Scale the dot product
        for (int j = 0; j < QK->rows; j++) {
            for (int k = 0; k < QK->cols; k++) {
                QK->data[j * QK->cols + k] /= sqrt(head_dim);
            }
        }

        // Apply softmax
        softmax(QK);

        // Multiply by V
        attention_output_h[i] = matmul(QK, VWV);

        // Free intermediate matrices
        free_matrix(QWQ);
        free_matrix(KWK);
        free_matrix(VWV);
        free_matrix(K_T);
        free_matrix(QK);
    }

    // Concatenate the attention outputs
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            int head = j / head_dim;
            int index = j % head_dim;
            attention_output->data[i * input->cols + j] = attention_output_h[head]->data[i * head_dim + index];
        }
    }

    // Multiply by output weight matrix
    Matrix_t *final_output = matmul(attention_output, &layer->W_O);

    // Add residual connection and apply layer normalization
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            final_output->data[i * final_output->cols + j] += input->data[i * input->cols + j];
        }
    }

    layer_normalization(final_output, layer->gamma1, layer->beta1);

    // Free all the data
    for (int i = 0; i < num_heads; i++) {
        free_matrix(Q_heads[i]);
        free_matrix(K_heads[i]);
        free_matrix(V_heads[i]);
        free_matrix(attention_output_h[i]);
    }
    free(Q_heads);
    free(K_heads);
    free(V_heads);
    free(attention_output_h);
    free_matrix(attention_output);

    // Return final output
    return final_output;
}

void attention_heads_v2(Transformer_t *transformer, Matrix_t *input) {
    int num_heads = transformer->num_heads;
    TransformerLayer_t *layer = transformer->layers;
    int head_dim = input->cols / num_heads;

    Matrix_t **Q_heads = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t **K_heads = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t **V_heads = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t **attention_output_h = (Matrix_t **)malloc(num_heads * sizeof(Matrix_t *));
    Matrix_t *attention_output = create_matrix(input->rows, input->cols, 64);

    for (int i = 0; i < num_heads; i++) {
        Q_heads[i] = create_matrix(input->rows, head_dim, 64);
        K_heads[i] = create_matrix(input->rows, head_dim, 64);
        V_heads[i] = create_matrix(input->rows, head_dim, 64);

        for (int j = 0; j < input->rows; j++) {
            for (int k = 0; k < head_dim; k++) {
                Q_heads[i]->data[j * head_dim + k] = input->data[j * input->cols + i * head_dim + k];
                K_heads[i]->data[j * head_dim + k] = input->data[j * input->cols + i * head_dim + k];
                V_heads[i]->data[j * head_dim + k] = input->data[j * input->cols + i * head_dim + k];
            }
        }

        // Compute attention for each head
        Matrix_t *QWQ = matmul(Q_heads[i], layer->W_Q);
        Matrix_t *KWK = matmul(K_heads[i], layer->W_K);
        Matrix_t *VWV = matmul(V_heads[i], layer->W_V);

        // Scaled dot-product attention
        Matrix_t *K_T = transpose(KWK);
        Matrix_t *QK = matmul(QWQ, K_T);

        // Scale the dot product
        for (int j = 0; j < QK->rows; j++) {
            for (int k = 0; k < QK->cols; k++) {
                QK->data[j * QK->cols + k] /= sqrt((float)head_dim);
            }
        }

        // Apply softmax
        softmax(QK);

        // Multiply by V
        attention_output_h[i] = matmul(QK, VWV);

        // Free intermediate matrices
        free_matrix(QWQ);
        free_matrix(KWK);
        free_matrix(VWV);
        free_matrix(K_T);
        free_matrix(QK);
    }

    // Concatenate the attention outputs
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            int head = j / head_dim;
            int index = j % head_dim;
            attention_output->data[i * input->cols + j] = attention_output_h[head]->data[i * head_dim + index];
        }
    }

    // Multiply by output weight matrix
    Matrix_t *final_output = matmul(attention_output, layer->W_O);

    // Assign the final output to input
    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            input->data[i * input->cols + j] = final_output->data[i * final_output->cols + j];
        }
    }

    // Free all the data
    for (int i = 0; i < num_heads; i++) {
        free_matrix(Q_heads[i]);
        free_matrix(K_heads[i]);
        free_matrix(V_heads[i]);
        free_matrix(attention_output_h[i]);
    }
    free(Q_heads);
    free(K_heads);
    free(V_heads);
    free(attention_output_h);
    free_matrix(final_output);
    free_matrix(attention_output);
}

Matrix_t* linear_transformer(TransformerLayer_t *layer, Matrix_t *input) {
    // Implement linear transformer calculation
    /*
        1. Multiply input by W1
        2. Add bias b1
        3. Apply ReLU activation
        4. Multiply by W2
        5. Add bias b2
        6. Add residual connection
        7. Apply layer normalization
        8. Return output
    */

    // Step 1: Multiply input by W1
    Matrix_t *output = matmul(input, layer->W1);

    // Step 2: Add bias b1
    for (int j = 0; j < output->rows; j++) {
        for (int k = 0; k < output->cols; k++) {
            output->data[j * output->cols + k] += layer->b1->data[k];
        }
    }

    // Step 3: Apply ReLU activation
    relu(output);

    // Step 4: Multiply by W2
    Matrix_t *output2 = matmul(output, layer->W2);

    // Free intermediate matrix
    free_matrix(output);

    // Step 5: Add bias b2
    for (int j = 0; j < output2->rows; j++) {
        for (int k = 0; k < output2->cols; k++) {
            output2->data[j * output2->cols + k] += layer->b2->data[k];
        }
    }

    // Step 6: Add residual connection
    for (int j = 0; j < output2->rows; j++) {
        for (int k = 0; k < output2->cols; k++) {
            output2->data[j * output2->cols + k] += input->data[j * input->cols + k];
        }
    }

    // Step 7: Apply layer normalization
    layer_normalization(output2, layer->gamma2, layer->beta2);

    // Step 8: Return output
    return output2;
}

Matrix_t* transformer_layer(TransformerLayer_t *layer, Matrix_t *input) {
    // Multi-Head Self-Attention
    Matrix_t *attention_output = attention_heads(layer, input);

    // Position-Wise Feed-Forward Network (FFN)
    Matrix_t *ffn_output = linear_transformer(layer, attention_output);

    // Free intermediate matrices
    free_matrix(attention_output);

    // Return the final output
    return ffn_output;
}

Matrix_t* apply_transformer_layers(Transformer_t *transformer, Matrix_t *input) {
    Matrix_t *current_input = input;

    for (int i = 0; i < transformer->num_layers; i++) {
        current_input = transformer_layer(&transformer->layers[i], current_input);
    }

    return current_input;
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
