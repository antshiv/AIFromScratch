#include <stdio.h>
#include <stdlib.h>
#include "hdf5_utils.h"
#include "deep_neural_network.h"

#define DATASET_TRAIN_SET_X "train_set_x"
#define DATASET_TRAIN_SET_Y "train_set_y"
#define DATASET_TEST_SET_X "test_set_x"
#define DATASET_TEST_SET_Y "test_set_y"
    
const char *train_file_name = "../dataset/train_catvnoncat.h5";
const char *test_file_name = "../dataset/test_catvnoncat.h5";

int main()
{
   dataset_info_t datasets[4]; 
   dataset_t dataset_x_train, dataset_y_train, dataset_x_test, dataset_y_test;

    // Initialize dataset information
    printf("Initializing dataset information\n");
    datasets[0] = (dataset_info_t){DATASET_TRAIN_SET_X, TRAIN, &dataset_x_train, train_file_name};
    datasets[1] = (dataset_info_t){DATASET_TRAIN_SET_Y, TRAIN, &dataset_y_train, train_file_name};
    datasets[2] = (dataset_info_t){DATASET_TEST_SET_X, TEST, &dataset_x_test, test_file_name};
    datasets[3] = (dataset_info_t){DATASET_TEST_SET_Y, TEST, &dataset_y_test, test_file_name};


    for (int i = 0; i < 4; i++)
    {
        printf("Populating dataset: %s %d %s \n", datasets[i].name, datasets[i].type, datasets[i].file_name);
        populate_dataset(datasets[i].dataset, datasets[i].name, datasets[i].type, datasets[i].file_name);
        loaddata(datasets[i].dataset);
    }

    layerdims_t layer_dims;
    layer_dims.layer_dims = (int *)malloc(4 * sizeof(int));
    layer_dims.layer_dims[0] = 12288;
    layer_dims.layer_dims[1] = 20;
    layer_dims.layer_dims[2] = 7;
    layer_dims.layer_dims[3] = 5;
    layer_dims.layer_dims[4] = 1;
    layer_dims.L = 5;


    //dnn(&dataset_x_train, &dataset_y_train, &layer_dims, 0.0075, 2500);
    dnn_v2(&dataset_x_train, &dataset_y_train, 0.01, 10);

    printf("Freeing layer dims\n");
    free(layer_dims.layer_dims);
   
    printf("Closing datasets\n"); 
    for (int i = 0; i < 4; i++)
    {
        closedata(datasets[i].dataset);
    }

    return 0;

}