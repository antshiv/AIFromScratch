/*
    This file contains code to train a logisitc regression neural network
    to classify the MNIST dataset. The MNIST dataset is a dataset of 28x28
    pixel images of handwritten digits. The dataset contains 60,000 training
    images and 10,000 testing images. The images are labeled with the correct
    digit they represent. The neural network is trained using the backpropagation
    algorithm. The neural network is trained using the training images and labels.

    The neural network is trained using the following steps:
    1. Load the training images and labels from the MNIST dataset.
    2. Initialize the weights and biases of the neural network.
    3. Train the neural network using the training images and labels.
    4. Test the neural network using the testing images and labels.
    5. Calculate the accuracy of the neural network on the testing images.
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn.h"
#include <string.h>

#define PARAMETER_SIZE 784

void normalize_dataset(dataset_t *dataset, size_t size) {
    printf("Normalizing the dataset\n");
    for (size_t i = 0; i < size; i++) {
        ((unsigned char *)dataset->data)[i] = ((unsigned char *)dataset->data)[i] / 255.0;
    }
}

void initialze_weights(weights_t *weights, int size) { 
    printf("Initialzing weights and biases\n");
    weights->w = calloc(size, sizeof(float));
    weights->b = 0;
}

void sigmoid(float *A, int size) {
    printf("Calculating sigmoid\n");
    for (int i = 0; i < size; i++) {
        A[i] = 1 / (1 + exp(-A[i]));
    }
    printf("Sigmoid of A[0] %f\n", A[0]); 
}

void pridict(float *W, float b, int *X) {
    printf("Predicting output\n");
    printf("W[0] %f, b %f, X[0] %d\n", W[0], b, X[0]);
}

// Calculate the cost function
float calculate_cost(float *A, dataset_t *Y_train, int m) {
    float cost = 0;
    for (int i = 0; i < m; i++) {
        cost += -(((long long *)Y_train->data)[i] * log(A[i]) + (1 - ((long long *)Y_train->data)[i]) * log(1 - A[i]));
    }
    return cost / m;
}

// Forward propagation
void forward_propagate(float *A, weights_t weights, dataset_t *X_train, int m, int size) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = weights.w[j] * ((unsigned char *)X_train->data)[i * size + j] + weights.b;
        }
    }
    sigmoid(A, m * size);
}

// Backward propagation
void backward_propagate(gradients_t *gradients, float *A, dataset_t *X_train, dataset_t *Y_train, int m, int size) {
    float *dz = calloc(m * size, sizeof(float));
    if (dz == NULL) {
        fprintf(stderr, "Error allocating memory for dz\n"); 
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < size; j++) {
            dz[i * size + j] = A[i * size + j] - ((long long *)Y_train->data)[i];
        }
    }

    // Initialize gradients
    gradients->dw = calloc(size, sizeof(float));
    if (gradients->dw == NULL) {
        fprintf(stderr, "Error allocating memory for gradients->dw\n");
        free(dz);
        exit(EXIT_FAILURE);
    }
    gradients->db = 0.0;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < size; j++) {
            gradients->dw[j] += dz[i * size + j] * ((unsigned char *)X_train->data)[i * size + j];
        }
        gradients->db += dz[i * size];
    }

    for (int j = 0; j < size; j++) {
        gradients->dw[j] /= m;
    }
    gradients->db /= m;

    free(dz);
}

// Main backpropagation function
void backpropagate(gradients_t *gradients, weights_t weights, dataset_t *X_train, dataset_t *Y_train, bool print) {
    printf("Propagating forward and backward\n");

    // Number of training examples
    int m = X_train->dims[0];
    int size = X_train->dims[1] * X_train->dims[2] * X_train->dims[3];

    float *A = calloc(m * size, sizeof(float));
    if (A == NULL) {
        fprintf(stderr, "Error allocating memory for A\n");
        exit(EXIT_FAILURE);
    }

    // Forward propagation
    forward_propagate(A, weights, X_train, m, size);

    // Calculate the cost
    float cost = calculate_cost(A, Y_train, m);
    if (print) {
        printf("Cost is %f\n", cost);
    }

    // Backward propagation
    backward_propagate(gradients, A, X_train, Y_train, m, size);

    if (print) {
        printf("Gradients dw[0] %f and db %f\n", gradients->dw[0], gradients->db);
    }

    free(A);
}

void optimize(dataset_t *X_train, dataset_t *Y_train, weights_t weights, int epoch, float learning_rate, bool print) {
    gradients_t gradients;
    printf("Optimizing weights and biases\n");
    //printf("LEarning rate %f\n", learning_rate);
    
    // Number of training examples
    int size = X_train->dims[1] * X_train->dims[2] * X_train->dims[3];
    
    float *dw;
    float db;
    
    gradients.dw = calloc(size, sizeof(float));
    if (!gradients.dw) {
        fprintf(stderr, "Error allocating memory\n");
        exit(EXIT_FAILURE);
    }

    printf("starting weights %f, %f\n", weights.w[0], weights.b);
    
    for (int i=0; i < epoch; i++) {
        printf("\n\nEpoch %d\n", i);
        
        backpropagate(&gradients, weights, X_train, Y_train, print);
        
        dw = gradients.dw;
        db = gradients.db;
        
        printf("Gradients dw[0] %f and db %f \n", dw[0], db);
        
        for (hsize_t j = 0; j < X_train->dims[1] * X_train->dims[2] * X_train->dims[3]; j++) {
            weights.w[j] -= learning_rate * dw[j];
        }
        
        weights.b -= learning_rate * db;
        
        printf("Updated weights %f, %f\n", weights.w[0], weights.b);
    }
    free(gradients.dw);
}

// Function to save weights to a binary file
void save_weights_binary(const char *filename, weights_t *weights, size_t num_weights) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    fwrite(weights->w, sizeof(float), num_weights, file);
    fclose(file);
}

// Function to save weights to a text file
void save_weights_text(const char *filename, weights_t *weights, size_t num_weights) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    for (size_t i = 0; i < num_weights; ++i) {
        fprintf(file, "%f\n", weights->w[i]);
    }
    fclose(file);
}

void model(dataset_t *X_train, dataset_t *Y_train, dataset_t *X_test, dataset_t *Y_test, int epoch, float learning_rate, bool print) {
    if (print)
        printf("Modeling the neural network\n");
    weights_t weights;
    normalize_dataset(X_train, X_train->dims[0] * X_train->dims[1] * X_train->dims[2] * X_train->dims[3]);
    int parameter_size = X_train->dims[1] * X_train->dims[2] * X_train->dims[3];
    if (print)
        printf("The number of parameters is %d \n", parameter_size);

    initialze_weights(&weights , parameter_size);
    if (print) 
        printf("Initial weights and biases are %f, %f\n", weights.w[0], weights.b);

    optimize(X_train, Y_train, weights, epoch, learning_rate, print);
     size_t num_weights = X_train->dims[1] * X_train->dims[2] * X_train->dims[3];

    save_weights_binary("weights.bin", &weights, num_weights);
    save_weights_text("weights.txt", &weights, num_weights);
    //pridict(W, b, X_test);
    printf("Modeling done\n");
    //forward_propagate(weights.w, weights, X_test, X_test->dims[0], parameter_size);
    //printf("Predicted output %f\n", weights.w[0]);
    //printf("Comparing with actual output %ld\n", ((long *)Y_test->data)[0]);
    free(weights.w);
    return;
}



/*
int main() {
    printf("Hello World! \n");
    for (int i = 0; i < 10; i++) {
        printf("Value of %c %d\n",data[i], a[i]);
    }
    float W[PARAMETER_SIZE];
    float b;
    initialze_weights(W, b, PARAMETER_SIZE);
    printf("Value of W[0] %f\n", W[0]);
}
*/