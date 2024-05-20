#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

#define TEST_FILE_NAME "test_catvnoncat.h5"
#define TRAIN_FILE_NAME "train_catvnoncat.h5"
#define DATASET_LIST_CLASSES "/list_classes"
#define DATASET_TEST_SET_X "/test_set_x"
#define DATASET_TEST_SET_Y "/test_set_y"

int main() {
    hid_t file_id, dataset_id, space_id, datatype_id;
    herr_t status;
    
    // Open the HDF5 file
    file_id = H5Fopen(TEST_FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error opening file: %s\n", TEST_FILE_NAME);
        return 1;
    }

    // Read the test_set_x dataset
    dataset_id = H5Dopen2(file_id, DATASET_TEST_SET_X, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", DATASET_TEST_SET_X);
        H5Fclose(file_id);
        return 1;
    }

    unsigned char test_set_x[50][64][64][3];  // Adjust dimensions accordingly
    status = H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, test_set_x);
    if (status < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", DATASET_TEST_SET_X);
    } else {
        printf("test_set_x[0][0][0]: %u\n", test_set_x[0][0][0][0]);  // Example output
    }
    H5Dclose(dataset_id);

    // Read the test_set_y dataset
    dataset_id = H5Dopen2(file_id, DATASET_TEST_SET_Y, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", DATASET_TEST_SET_Y);
        H5Fclose(file_id);
        return 1;
    }

    long long test_set_y[50];  // Assuming 64-bit integers
    status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, test_set_y);
    if (status < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", DATASET_TEST_SET_Y);
    } else {
        printf("test_set_y:\n");
        for (int i = 0; i < 50; i++) {
            printf("%lld ", test_set_y[i]);
        }
        printf("\n");
    }
    H5Dclose(dataset_id);

    // Close the HDF5 file
    H5Fclose(file_id);

    return 0;
}