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

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"

char data[10] = {'a','b','c','d','e','f','g','h','i','j'};
int a[10] = {1,2,3,4,5,6,7,8,9,10};
#define PARAMETER_SIZE 784

void initialze_weights(float *w, float b, int size) {
    printf("Initialzing weights and biases\n");
    for (int i = 0; i < size; i++) {
        w[i] = rand() % 100;
    }
    b = rand() % 100;
}

void sigmoid(int *z, int size, int *s) {
    printf("Calculating sigmoid\n");
    for (int i = 0; i < size; i++) {
        s[i] = 1 / (1 + exp(-z[i]));
    } 
}

void pridict(float *W, float b, int *X) {
    printf("Predicting output\n");
}

void propogate(int *X, int *Y, int *w, int b, int size, int *a, int *dw, int db, int *weight_size) {
    printf("Propogating forward and backward\n");
    int z[size];
    int s[size];
    int cost;
    int sum;
    int dw_sum;

    for (int i = 0; i < size; i++) {
        z[i] = w[i] * X[i] + b;
    }

    sigmoid(z, size, s);

    for (int i = 0; i < size; i++) {
        a[i] = s[i] - Y[i];
    }

    for (int i = 0; i < size; i++) {
        sum += Y[i]*log(s[i]) + (1-Y[i])*log(1-s[i]); 
    }

    cost = -sum/size;

    for (int i = 0; i < size; i++) {
        int diff = (s[i] - Y[i]);
        dw[i] = X[i] * diff;
        dw[i] = dw[i]/size;
        db += diff;
    }

}

void optimize(int *X_train, int *Y_train, int epoch, int learning_rate, int print) {
    printf("Optimizing weights and biases\n");
}

void model(int *X_train, int *Y_train, int *X_test, int *Y_test, int epoch, int learning_rate, int print) {
    float W[PARAMETER_SIZE];
    float b;
    initialze_weights(W, b, PARAMETER_SIZE);
    optimize(X_train, Y_train, epoch, learning_rate, print);
    pridict(W, b, X_test);
}

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