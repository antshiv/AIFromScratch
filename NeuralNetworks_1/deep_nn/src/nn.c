#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "hdf5_utils.h"
#include "dnn.h"

// Function to calculate the sigmoid of a number
void sigmoid(ActivationResult *result)
{
    for (int i = 0; i < result->size; i++)
    {
        result->A[i] = 1 / (1 + exp(-result->A[i]));
    }
}

// Function to allocate a 2D array (matrix)
double **allocate_matrix(int rows, int cols)
{
    double **matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (double *)malloc(cols * sizeof(double));
    }
    return matrix;
}

// Function to allocate a 1D array (vector)
double *allocate_vector(int size)
{
    return (double *)malloc(size * sizeof(double));
}

// Function to initialize parameters
LayerParameters *initialize_parameters_deep(layerdims_t *layerdims)
{
    LayerParameters *parameters = (LayerParameters *)malloc((L - 1) * sizeof(LayerParameters));
    srand(1); // Seed the random number generator

    for (int l = 1; l < layerdims->L; l++)
    {
        int rows = layerdims->layer_dims[l];
        int cols = layerdims->layer_dims[l - 1];

        // Allocate memory for W and b
        parameters[l - 1].W = allocate_matrix(rows, cols);
        parameters[l - 1].b = allocate_vector(rows);

        // Initialize W with random values scaled by sqrt(1 / cols)
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                parameters[l - 1].W[i][j] = ((double)rand() / RAND_MAX) / sqrt((double)cols);
            }
        }

        // Initialize b with zeros
        for (int i = 0; i < rows; i++)
        {
            parameters[l - 1].b[i] = 0.0;
        }

        // Assert the shapes (this is optional in C and for debugging purposes)
        assert(parameters[l - 1].W != NULL);
        assert(parameters[l - 1].b != NULL);
    }
    return parameters;
}

// Function to free the allocated memory for parameters
void free_parameters(LayerParameters *parameters, int L)
{
    for (int l = 0; l < L - 1; l++)
    {
        for (int i = 0; i < layer_dims[l + 1]; i++)
        {
            free(parameters[l].W[i]);
        }
        free(parameters[l].W);
        free(parameters[l].b);
    }
    free(parameters);
}

void model_forward()
{
}

void compute_cost()
{
}

void model_backward()
{
}

void update_parameters()
{
}

void dnn(dataset_t *X_train, dataset_t *Y_train, layerdims_t *layer_dims, float learning_rate, int num_iterations)
{
    // Example usage
    int layer_dims[] = {5, 4, 3}; // Example layer dimensions
    int L = sizeof(layer_dims) / sizeof(layer_dims[0]);

    LayerParameters *parameters = initialize_parameters_deep(layer_dims);

    // Free the allocated memory
    free_parameters(parameters, L);

    return 0;
}