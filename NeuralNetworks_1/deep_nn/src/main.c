#include <stdio.h>
#include <stdlib.h>
#include "hdf5_utils.h"
#include "deep_neural_network.h"

#define DATASET_TRAIN_SET_X "train_set_x"
#define DATASET_TRAIN_SET_Y "train_set_y"
#define DATASET_TEST_SET_X "test_set_x"
#define DATASET_TEST_SET_Y "test_set_y"

dataset_info_t datasets[4] = {
    {"DATASET_TRAIN_SET_X", TRAIN},
    {"DATASET_TRAIN_SET_Y", TRAIN},
    {"DATASET_TEST_SET_X", TEST},
    {"DATASET_TEST_SET_Y", TEST}};

void main()
{
    dataset_t dataset_train_x, dataset_train_y, dataset_test_x, dataset_test_y;
    for (int i = 0; i < 4; i++)
    {
        populate_dataset(&datasets[i], datasets[i].name, datasets[i].type);
        loaddata(&datasets[i]);
    }
}