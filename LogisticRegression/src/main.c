#include <stdio.h>
#include <stdlib.h>
#include "hdf5_utils.h"

#define DATASET_TRAIN_SET_X "train_set_x"
#define DATASET_TRAIN_SET_Y "train_set_y"

int main() {
    // File and dataset names
    const char *train_file_name = "dataset/train_catvnoncat.h5";
    const char *test_file_name = "dataset/test_catvnoncat.h5";

    // Open the HDF5 files
    hid_t train_file_id = open_file(train_file_name);
    if (train_file_id < 0) return 1;

    hid_t test_file_id = open_file(test_file_name);
    if (test_file_id < 0) {
        close_file(train_file_id);
        return 1;
    }

        // Read the train_set_x dataset
    hid_t datatype;
    hsize_t *dims;
    int ndims;
    unsigned char *train_set_x = (unsigned char *)read_dataset(train_file_id, "train_set_x", &datatype, &dims, &ndims);
    if (!train_set_x) {
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }
    printf("The number of train_set_x dimensions is %d \n", ndims);
    for (int i=0; i < ndims; i++) {
        printf("Value of dime [%d] is %ld \n",i, dims[i]); 
    }
    uint8_t r,g,b;
    get_rgb_values(train_set_x, dims[1], dims[2], dims[3], 200, 56, 59, &r, &g, &b);
    printf("The values of r = %d, g = %d, b = %d \n", r,g,b);

    // Read the train_set_y dataset
    long long *train_set_y = (long long *)read_dataset(train_file_id, "train_set_y", &datatype, &dims, &ndims);
    if (!train_set_y) {
        free(train_set_x);
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

    // Read the test_set_x dataset
    unsigned char *test_set_x = (unsigned char *)read_dataset(test_file_id, "test_set_x", &datatype, &dims, &ndims);
    if (!test_set_x) {
        free(train_set_x);
        free(train_set_y);
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

    // Read the test_set_y dataset
    long long *test_set_y = (long long *)read_dataset(test_file_id, "test_set_y", &datatype, &dims, &ndims);
    if (!test_set_y) {
        free(train_set_x);
        free(train_set_y);
        free(test_set_x);
        close_file(train_file_id);
        close_file(test_file_id);
        return 1;
    }

    // Cleanup
    free(train_set_x);
    free(train_set_y);
    free(test_set_x);
    free(test_set_y);
    free(dims);
    close_file(train_file_id);
    close_file(test_file_id);
    return 0;
}