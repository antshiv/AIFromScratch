#include <stdio.h>
#include <stdlib.h>
#include "hdf5_utils.h"

#define DATASET_TRAIN_SET_X "train_set_x"
#define DATASET_TRAIN_SET_Y "train_set_y"

int main()
{
    // File and dataset names
    const char *train_file_name = "dataset/train_catvnoncat.h5";
    const char *test_file_name = "dataset/test_catvnoncat.h5";

    // Open the HDF5 files
    hid_t train_file_id = open_file(train_file_name);
    if (train_file_id < 0)
        return 1;

    hid_t test_file_id = open_file(test_file_name);
    if (test_file_id < 0)
    {
        close_file(train_file_id);
        return 1;
    }

    // Read the train_set_x dataset
    hid_t datatype;
    hsize_t *dims_x_train, *dims_y_train, *dims_x_test, *dims_y_test;
    int ndims_x_train, ndims_y_train, ndims_x_test, ndims_y_test;
    unsigned char *train_set_x = (unsigned char *)read_dataset(train_file_id, "train_set_x", &datatype, &dims_x_train, &ndims_x_train);
    if (!train_set_x)
    {
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

#ifdef DEBUG
    printf("The number of train_set_x dimensions is %d \n", ndims_x_train);
    for (int i = 0; i < ndims_x_train; i++)
    {
        printf("Value of dime [%d] is %ld \n", i, dims_x_train[i]);
    }
    uint8_t r, g, b;
    get_rgb_values(train_set_x, dims_x_train[1], dims_x_train[2], dims_x_train[3], 200, 63, 59, &r, &g, &b);
    printf("The values of r = %d, g = %d, b = %d \n", r, g, b);
#endif

    // Normalize the dataset

    // Read the train_set_y dataset
    long long *train_set_y = (long long *)read_dataset(train_file_id, "train_set_y", &datatype, &dims_y_train, &ndims_y_train);
    if (!train_set_y)
    {
        free(train_set_x);
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

#ifdef DEBUG
    for (int i = 0; i < dims_y_train[0]; i++)
    {
        printf("Value of train_set_y [%d] is %lld \n", i, train_set_y[i]);
    }
#endif

    // Read the test_set_x dataset
    unsigned char *test_set_x = (unsigned char *)read_dataset(test_file_id, "test_set_x", &datatype, &dims_x_test, &ndims_x_test);
    if (!test_set_x)
    {
        free(train_set_x);
        free(train_set_y);
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

    // Read the test_set_y dataset
    long long *test_set_y = (long long *)read_dataset(test_file_id, "test_set_y", &datatype, &dims_y_test, &ndims_y_test);
    if (!test_set_y)
    {
        free(train_set_x);
        free(train_set_y);
        free(test_set_x);
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

    printf("The number of training Examples is %ld \n", dims_x_train[0]);
    printf("The number of test Examples is %ld \n", dims_x_test[0]);
    printf("The height and width of each image is  %ld, %ld \n", dims_x_train[1], dims_x_train[2]);
    printf("The number of channels is %ld \n", dims_x_train[3]);
    printf("Each image is of size %ld \n", dims_x_train[1] * dims_x_train[2] * dims_x_train[3]);
    printf("Train set x shape is %ld, %ld, %ld, %ld \n", dims_x_train[0], dims_x_train[1], dims_x_train[2], dims_x_train[3]);
    printf("Train set y shape is %ld \n", dims_y_train[0]);
    printf("Test set x shape is %ld, %ld, %ld, %ld \n", dims_x_test[0], dims_x_test[1], dims_x_test[2], dims_x_test[3]);
    printf("Test set y shape is %ld \n", dims_y_test[0]);

    /* Let us train the neural network */
    //void *nn = train_nn(train_set_x, train_set_y, num_dims_x_train, dims_x_train, num_dims_y_train, dims_y_train, 0.01, 1000);

    // Cleanup
    free(train_set_x);
    free(train_set_y);
    free(test_set_x);
    free(test_set_y);
    free(dims_x_train);
    free(dims_y_train);
    free(dims_x_test);
    free(dims_y_test);
    close_file(train_file_id);
    close_file(test_file_id);
    return 0;
}