#include "attention.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

MAtrix_t init_matrix(int rows, int cols)
{
    Matrix_t mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float *)malloc(rows * cols * sizeof(float));
    return mat;
}

void free_matrix(Matrix_t *mat)
{
    free(mat->data);
}

void print_matrix(Matrix_t *mat)
{
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            printf("%f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
}

Matrix_t matmul(Matrix_t *A, Matrix_t *B)
{
    Matrix_t C = init_matrix(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < B->cols; j++)
        {
            C.data[i * C.cols + j] = 0;
            for (int k = 0; k < A->cols; k++)
            {
                C.data[i * C.cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }
    return C;
}

Matrix_t transpose(const Matrix_t *mat)
{
    Matrix_t transposed = init_matrix(mat->cols, mat->rows);
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            transposed.data[j * transposed.cols + i] = mat->data[i * mat->cols + j];
        }
    }
    return transposed;
}

void softmax(const Matrix_t *mat)
{
    // 1st step: Find the maximum value in the matrix
    float max = mat->data[0];
    for (int i = 1; i < mat->rows * mat->cols; i++)
    {
        if (mat->data[i] > max)
        {
            max = mat->data[i];
        }
    }

    // 2nd step: Subtract the maximum value from all the elements
    //           exponentiate the result
    //           sum all the elements
    float sum = 0;
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->data[i] = exp(mat->data[i] - max);
        sum += mat->data[i];
    }

    // 3rd step: Divide all the elements by the sum
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->data[i] /= sum;
    }
}

Matrix_t attention_forward(const Matrix_t *Q, const Matrix_t *K, const Matrix_t *V)
{
    Matrix_t K_T = transpose(K);
    Matrix_t QK = matmul(Q, &K_T);
    softmax(&QK);
    Matrix_t attention_output = matmul(&QK, V);
    free_matrix(K_T);
    free_matrix(QK);
    return attention_output;
}

Matrix_t multi_head_attendtion_forward(const Matrix_t *Q, const Matrix_t *K, const Matrix_t *V, const Matrix_t *W_Q, const Matrix_t *W_K, const Matrix_t *W_V, const Matrix_t *W_O, int num_heads)
{
    int head_dim = Q->cols / num_heads;
    int batch_size = Q->rows;

    Matrix_t concantenated = init_matrix(batch_size, Q->cols);

    for (int h = 0; h < num_heads; h++) {
        // Split the Q, K, V matrices into num_heads
        Matrix_t Q_h = init_matrix(batch_size, head_dim);
        Matrix_t K_h = init_matrix(batch_size, head_dim);
        Matrix_t V_h = init_matrix(batch_size, head_dim);

        for  (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q_h.data[i * head_dim + j] = Q->data[i * Q->cols + h * head_dim + j];
                K_h.data[i * head_dim + j] = K->data[i * K->cols + h * head_dim + j];
                V_h.data[i * head_dim + j] = V->data[i * V->cols + h * head_dim + j];
            }
        }

        // Compute attentuion for each head
        Matrix_t attention_output_h = attention_forward(&Q_h, &K_h, &V_h);

        // Concatenate the attention outputs
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < head_dim; j++) {
                concantenated.data[i * Q->cols + h * head_dim + j] = attention_output_h.data[i * head_dim + j];
            }
        }

        free_matrix(&Q_h);
        free_matrix(&K_h);
        free_matrix(&V_h);
    }

    // Project the concatenated matrix
    Matrix_t output = matmul(&concantenated, W_O);
    free_matrix(&concantenated);
    return output;
}

// add matrix for residual connection
void add_matrices(const Matrix_t *A, const Matrix_t *B, Matrix_t *C)
{
    if (A->rows != B->rows || A->cols != B->cols || A->rows != C->rows || A->cols != C->cols)
    {
        printf("Matrix dimensions do not match\n");
        exit(1);
    }
    for (int i =0; i < A->rows * A->cols; i++)
    {
        C->data[i] = A->data[i] + B->data[i];
    }
}

// Layer Normalization 
void layer_norm(const Matrix_t *X, const Matrix_t *gamma, const Matrix_t *beta, Matrix_t *Y)
{
    if (X->cols != gamma->cols || X->cols != beta->cols || X->rows != Y->rows || X->cols != Y->cols)
    {
        printf("Matrix dimensions do not match\n");
        exit(1);
    }

    for (int i = 0; i < X->rows; i++)
    {
        float mean = 0;
        float variance = 0;
        for (int j = 0; j < X->cols; j++)
        {
            mean += X->data[i * X->cols + j];
        }
        mean /= X->cols;

        for (int j = 0; j < X->cols; j++)
        {
            variance += pow(X->data[i * X->cols + j] - mean, 2);
        }
        variance /= X->cols;

        for (int j = 0; j < X->cols; j++)
        {
            Y->data[i * X->cols + j] = (X->data[i * X->cols + j] - mean) / sqrt(variance + 1e-8);
            // Adding gama and beta to the normalized matrix. Easier to read when divided into two steps
            Y->data[i * X->cols + j] = gamma->data[j] * Y->data[i * X->cols + j] + beta->data[j];
        }
    }    
}
