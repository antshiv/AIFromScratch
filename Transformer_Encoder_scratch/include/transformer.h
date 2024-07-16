#include <stdio.h>

typedef struct Matrix
{
    int rows;      // Number of rows
    int cols;      // Number of columns
    int stride;    // Stride (number of elements between consecutive rows in memory)
    int alignment; // Alignment (for memory alignment purposes)
    float *data;   // Pointer to matrix data
} Matrix_t;

typedef struct TransformerLayer
{
    /* data */
    Matrix_t W_Q, W_K, W_V, W_O;
    Matrix_t gamma1, beta1, gamma2, beta2; // Layer normalization parameters
    Matrix_t W1, b1, W2, b2;
    // Adam optimizer parameters (gradients)
    Matrix_t d_W_Q, d_W_K, d_W_V, d_W_O;
    Matrix_t d_gamma1, d_beta1, d_gamma2, d_beta2;
    Matrix_t d_W1, d_b1, d_W2, d_b2;
} TransformerLayer_t;

typedef struct Transformer
{
    /* data */
    int num_layers;
    TransformerLayer_t *layers;
    int num_heads;
    Matrix_t *d_input;
    Matrix_t *d_output;
    int mini_batch_size;
} Transformer_t;

// Function declarations
Matrix_t create_matrix(int rows, int cols, int alignment);
void free_matrix(Matrix_t *matrix);
void xavier_init(Matrix_t *matrix);
void position_encoding(Matrix *matrix);
void attention_head(TransformerLayer_t *layer, Matrix_t *input);
void linear_transformer(TransformerLayer_t *layer, Matrix_t *input);
void layer_norm_residual(TransformerLayer_t *layer, Matrix_t *input, Matrix_t *output);
void ner_classification_layer(Transformer_t *transformer, Matrix_t *input);
void backpropagation(Transformer_t *transformer, Matrix_t *input, Matrix_t *output);
void calculate_loss(Matrix_t *predictions, Matrix_t *labels);
void adam_optimizer_update(Transformer_t *transformer);