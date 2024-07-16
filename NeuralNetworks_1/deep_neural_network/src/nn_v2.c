#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "hdf5_utils.h"
#include "deep_neural_network.h"
#include <string.h>

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float relu(float x)
{
    return fmaxf(0.0f, x);
}

float sigmoid_derivative(float x)
{
    return x * (1 - x);
}

float relu_derivative(float x)
{
    return (x > 0) ? 1 : 0;
}

void forward_propagation(NeuralNetwork_t *nn, float *X)
{
    for (int i = 0; i < nn->num_layers; i++)
    {
        Layer_t *layer = &nn->layers[i];
        float *input = (i == 0) ? X : nn->layers[i - 1].activations;
        int input_size = layer->prev_size;

        // Matrix multiplication and adding bias
        for (int j = 0; j < layer->size; j++)
        {
            layer->z[j] = layer->biases[j];
            for (int k = 0; k < input_size; k++)
            {
                layer->z[j] += layer->weights[j * input_size + k] * input[k];
            }
            if (i == nn->num_layers - 1)
            {
                layer->activations[j] = sigmoid(layer->z[j]);
            }
            else
            {
                layer->activations[j] = relu(layer->z[j]);
            }
        }
    }
}

float compute_loss(float *final_activation, float *Y, int output_size)
{
    float loss = 0.0;
    for (int i = 0; i < output_size; i++)
    {
        // logistic cross entropy loss function
        loss -= Y[i] * logf(final_activation[i]) + (1.0f - Y[i]) * logf(1.0f - final_activation[i]);
    }
    return loss;
}

float d_loss(float a, float y)
{
    return a - y;
}

void compute_dZ_output(Layer_t *output_layer, float *Y, float *dZ)
{
    for (int i = 0; i < output_layer->size; i++)
    {
        dZ[i] = d_loss(output_layer->activations[i], Y[i]); // dz for logistic loss and sigmoid activation
        //printf("dZ[%d]: %f\n", i, dZ[i]);
    }
}

void compute_dZ_hidden(Layer_t *current_layer, Layer_t *next_layer, float *dZ_next, float *dZ_current)
{
    for (int i = 0; i < current_layer->size; i++)
    {
        dZ_current[i] = 0.0f;
        for (int j = 0; j < next_layer->size; j++)
        {
            dZ_current[i] += next_layer->weights[j * current_layer->size + i] * dZ_next[j];
        }
        // RELU activation
        dZ_current[i] *= relu_derivative(current_layer->z[i]);
    }
}

void compute_gradients(Layer_t *layer, float *input, float *dZ)
{
    int input_size = layer->prev_size;
    for (int i = 0; i < layer->size; i++)
    {
        layer->d_biases[i] = dZ[i];
        for (int j = 0; j < input_size; j++)
        {
            layer->d_weights[i * input_size + j] = dZ[i] * input[j];
        }
    }
}

void backward_propagation(NeuralNetwork_t *nn, float *X, float *Y)
{
    float *dZ = (float *)malloc(nn->layers[nn->num_layers - 1].size * sizeof(float));
    if (dZ == NULL)
    {
        printf("Memory allocation failed for dZ\n");
        exit(1);
    }

    // Calculate output layer gradients
    Layer_t *output_layer = &nn->layers[nn->num_layers - 1];
    compute_dZ_output(output_layer, Y, dZ);

    // Calculate gradients for the output layer
    Layer_t *prev_layer = &nn->layers[nn->num_layers - 2];
    compute_gradients(output_layer, prev_layer->activations, dZ);

    // Backpropagate through hidden layers
    for (int l = nn->num_layers - 2; l >= 0; l--)
    {
        Layer_t *current_layer = &nn->layers[l];
        Layer_t *next_layer = &nn->layers[l + 1];

        // Allocate memory for the current layer dZ
        float *dZ_current = (float *)malloc(current_layer->size * sizeof(float));
        if (dZ_current == NULL)
        {
            printf("Memory allocation failed for dZ_current\n");
            exit(1);
        }

        // Compute dZ for the current layer
        compute_dZ_hidden(current_layer, next_layer, dZ, dZ_current);

        // Get input to the current layer (either X for the first layer or activations from the previous layer)
        float *input = (l == 0) ? X : nn->layers[l - 1].activations;

        // Compute gradients for the current layer
        compute_gradients(current_layer, input, dZ_current);

        // Update dZ for the next iteration
        free(dZ);
        dZ = dZ_current;
    }
    // Free the final dZ
    free(dZ);
}

void initialize_layer(Layer_t *layer, int size, int prev_size)
{
    layer->size = size;
    layer->prev_size = prev_size;
    layer->weights = (float *)malloc(size * prev_size * sizeof(float));
    layer->biases = (float *)malloc(size * sizeof(float));
    layer->z = (float *)malloc(size * sizeof(float));
    layer->activations = (float *)malloc(size * sizeof(float));
    layer->d_weights = (float *)malloc(size * prev_size * sizeof(float));
    layer->d_biases = (float *)malloc(size * sizeof(float));

    // Initialize weights and biases
    for (int i = 0; i < size; i++)
    {
        layer->biases[i] = 0.0;
        for (int j = 0; j < prev_size; j++)
        {
            layer->weights[i * prev_size + j] = 0.01 * ((float)rand() / RAND_MAX);
        }
    }
}

NeuralNetwork_t *create_neural_network(int input_size, int *layer_size, int num_layers)
{
    NeuralNetwork_t *nn = (NeuralNetwork_t *)malloc(sizeof(NeuralNetwork_t));
    nn->num_layers = num_layers;
    nn->layers = (Layer_t *)malloc(num_layers * sizeof(Layer_t));

    for (int i = 0; i < num_layers; i++)
    {
        int prev_size = (i == 0) ? input_size : layer_size[i - 1];
        initialize_layer(&nn->layers[i], layer_size[i], prev_size);
    }
    return nn;
}

void initialize_gradients(NeuralNetwork_t *nn) {
    for (int l = 0; l < nn->num_layers; l++) {
        Layer_t *layer = &nn->layers[l];
        memset(layer->d_weights, 0, layer->size * layer->prev_size * sizeof(float));
        memset(layer->d_biases, 0, layer->size * sizeof(float));
    }
}

// Function to shuffle the dataset (optional, for better training)
void shuffle_dataset(float *X, float *Y, int num_samples, int input_size, int output_size) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap X
        for (int k = 0; k < input_size; k++) {
            float temp = X[i * input_size + k];
            X[i * input_size + k] = X[j * input_size + k];
            X[j * input_size + k] = temp;
        }
        // Swap Y
        for (int k = 0; k < output_size; k++) {
            float temp = Y[i * output_size + k];
            Y[i * output_size + k] = Y[j * output_size + k];
            Y[j * output_size + k] = temp;
        }
    }
}

void accumulate_gradients(NeuralNetwork_t *nn, NeuralNetwork_t *batch_gradients)
{
    for (int l = 0; l < nn->num_layers; l++)
    {
        Layer_t *layer = &nn->layers[l];
        Layer_t *batch_layer = &batch_gradients->layers[l];

        for (int i = 0; i < layer->size * layer->prev_size; i++)
        {
            layer->d_weights[i] += batch_layer->d_weights[i];
        }
        for (int i = 0; i < layer->size; i++)
        {
            layer->d_biases[i] += batch_layer->d_biases[i];
        }
    }
}

void average_gradients(NeuralNetwork_t *nn, int batch_size)
{
    for (int l = 0; l < nn->num_layers; l++)
    {
        Layer_t *layer = &nn->layers[l];
        for (int i = 0; i < layer->size * layer->prev_size; i++)
        {
            layer->d_weights[i] /= batch_size;
        }
        for (int i = 0; i < layer->size; i++)
        {
            layer->d_biases[i] /= batch_size;
        }
    }
}

// Binary Cross-Entropy Loss Function
float compute_cost(float *output_activations, float *Y, int num_samples) {
    float cost = 0.0;
    for (int i = 0; i < num_samples; i++) {
        float y_hat = sigmoid(output_activations[i]); // Predicted probability
        float y = Y[i];
        cost += y * log(y_hat) + (1 - y) * log(1 - y_hat);
    }
    return -cost / num_samples;
}

void update_parameters(NeuralNetwork_t *nn, float learning_rate)
{
    for (int l = 0; l < nn->num_layers; l++)
    {
        Layer_t *layer = &nn->layers[l];
        for (int i = 0; i < layer->size * layer->prev_size; i++)
        {
            layer->weights[i] -= learning_rate * layer->d_weights[i];
        }
        for (int i = 0; i < layer->size; i++)
        {
            layer->biases[i] -= learning_rate * layer->d_biases[i];
        }
    }
}

void save_parameters(NeuralNetwork_t *nn, const char *filename)
{
    FILE *fp;
    fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    for (int l = 0; l < nn->num_layers; l++)
    {
        Layer_t *layer = &nn->layers[l];
        fwrite(layer->weights, sizeof(float), layer->size * layer->prev_size, fp);
        fwrite(layer->biases, sizeof(float), layer->size, fp);
    }
    fclose(fp);
}

void save_parameters_txt(NeuralNetwork_t *nn, const char *filename)
{
    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    for (int l = 0; l < nn->num_layers; l++)
    {
        Layer_t *layer = &nn->layers[l];
        fprintf(fp, "Layer %d\n", l);
        fprintf(fp, "Weights\n");
        for (int i = 0; i < layer->size; i++)
        {
            for (int j = 0; j < layer->prev_size; j++)
            {
                fprintf(fp, "%f ", layer->weights[i * layer->prev_size + j]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "Biases\n");
        for (int i = 0; i < layer->size; i++)
        {
            fprintf(fp, "%f ", layer->biases[i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void dnn_v2(dataset_t *X_train, dataset_t *Y_train, float learning_rate, int num_iterations)
{
    int layers_size[] = {20, 7, 5, 1};
    int num_layers = sizeof(layers_size) / sizeof(layers_size[0]);
    int input_size = X_train->dims[1] * X_train->dims[2] * X_train->dims[3];
    int m = X_train->dims[0]; // Number of examples to train
    printf("Num samples %ld  - Input dataset dimensions: %ld x %ld x %ld\n", X_train->dims[0], X_train->dims[1], X_train->dims[2], X_train->dims[3]);
    printf("Output dataset dimensions: %ld \n", Y_train->dims[0]);
    printf("Learning rate: %f\n", learning_rate);
    printf("Number of iterations: %d\n", num_iterations);
    printf("Number of layers: %d\n", num_layers);
    printf("size of each layer: ");
    for (int i = 0; i < num_layers; i++)
    {
        printf("%d ", layers_size[i]);
    }
    printf("\n");

    NeuralNetwork_t *nn = create_neural_network(input_size, layers_size, num_layers);
    NeuralNetwork_t *batch_gradients = create_neural_network(input_size, layers_size, num_layers);

    int batch_size = 64;

    int num_batches = (m + batch_size - 1) / batch_size; // Calculate the number of batches
    int epochs = num_iterations / num_batches;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        // Optionally shuffle the dataset at the beginning of each epoch
        //shuffle_dataset(X_train->data, Y_train->data, m, input_size, 1);

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start = batch * batch_size;
            int end = start + batch_size;
            if (end > m)
            {
                end = m;
            }

            // Reset batch gradients
            initialize_gradients(batch_gradients);

            float batch_cost = 0.0f;

            for (int i = start; i < end; i++)
            {
                forward_propagation(nn, X_train->data + i * input_size);
                backward_propagation(nn, X_train->data + i * input_size, Y_train->data + i);
                accumulate_gradients(batch_gradients, nn);

                // Accumulate cost for the batch
                batch_cost += compute_cost(nn->layers[num_layers - 1].activations, Y_train->data + i, 1);
            }

            average_gradients(batch_gradients, end - start);
            update_parameters(nn, learning_rate);

            // Average cost for the batch
            batch_cost /= (end - start);

            //if (batch % 10 == 0)
            //{
                printf("Batch %d/%d\n", batch + 1, num_batches);
                printf("Cost after batch %d: %f\n", batch, batch_cost);
            //}
        }
        printf("----------------------\n");
        printf("\n");
    }

    // print parameters to file as binary and as txt
    save_parameters(nn, "parameters.bin");
    save_parameters_txt(nn, "parameters.txt");

    // free memory
    printf("\n----------------------\n");
    printf("Freeing memory\n");
    for (int i = 0; i < num_layers; i++)
    {
        printf("Freeing layer %d\n", i);
        free(nn->layers[i].weights);
        free(nn->layers[i].biases);
        free(nn->layers[i].z);
        free(nn->layers[i].activations);
        free(nn->layers[i].d_weights);
        free(nn->layers[i].d_biases);
    }
    free(nn->layers);
    free(nn);
}