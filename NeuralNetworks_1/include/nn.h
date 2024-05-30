/*
 * This file contains all the declaration to run a nn
 */

#include <hdf5.h>
#include <stdbool.h>

typedef struct weights
{
    float *w;
    float b;
} weights_t;

typedef struct gradients
{
    float *dw;
    float db;
} gradients_t;

void model(dataset_t *X_train, dataset_t *Y_train, dataset_t *X_test, dataset_t *Y_test, int epoch, float learning_rate, bool print);