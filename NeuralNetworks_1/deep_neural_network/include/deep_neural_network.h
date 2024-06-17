#include <stdio.h>
#include <stdlib.h>
#include "hdf5_utils.h"

// Structure to hold the parameters W and b
typedef struct {
    double** W;
    double* b;
} LayerParameters_t;

// Structure to hold the result and cache
typedef struct {
    double* A;
    double* cache;
    int size;
} ActivationResult_t;

typedef struct {
    double** da;  // Gradient of activation
    double* db;   // Gradient of bias
    double** dw; // Gradient of weights
} Gradient_t;


typedef struct layerdims {
    int* layer_dims;
    int L;
} layerdims_t;

void dnn(dataset_t *X_train, dataset_t *Y_train, layerdims_t *layer_dims, float learning_rate, int num_iterations);


typedef struct Layer {
    int size; // Number of nueron sin this layer
    int prev_size; // Number of neurons in the previous layer
    float *weights; // Weight matrix(size, prev_size)
    float *biases; // Bias vector (size)
    float *z; // Weighted sum vector (size) 
    float *activations; // Activation vector (size)
    float *d_weights; // Gradient of weights matrix (size, prev_size)
    float *d_biases; // Gradient of biases vector (size)

} Layer_t;

typedef struct NeuralNetwork {
    int num_layers;
    Layer_t *layers;
} NeuralNetwork_t; 

void dnn_v2(dataset_t *X_train, dataset_t *Y_train, float learning_rate , int num_iterations);