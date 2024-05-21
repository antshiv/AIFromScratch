#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

#define FILE_NAME "train_catvnoncat.h5"
#define DATASET_LIST_CLASSES "list_classes"
#define DATASET_TRAIN_SET_X "train_set_x"
#define DATASET_TRAIN_SET_Y "train_set_y"

int main() {
    hid_t file_id, dataset_id, datatype, dataspace;
    herr_t status;

    // Open the HDF5 file
    file_id = H5Fopen(FILE_NAME, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error opening file: %s\n", FILE_NAME);
        return 1;
    }

    hid_t h5tools_get_native_type(hid_t type)
{
hid_t p_type;
H5T_class_t type_class;

type_class = H5Tget_class(type);
if (type_class == H5T_BITFIELD)
    p_type = H5Tcopy(type);
else
    p_type = H5Tget_native_type(type, H5T_DIR_DEFAULT);

return(p_type);
}

hid_t type, native_type;


    // Read the list_classes dataset
    dataset_id = H5Dopen(file_id, DATASET_LIST_CLASSES, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", DATASET_LIST_CLASSES);
        H5Fclose(file_id);
        return 1;
    }

    datatype = H5Dget_type(dataset_id);
    dataspace = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);

    char (*list_classes)[7] = malloc(dims[0] * sizeof(*list_classes));  // Allocate memory
    if (list_classes == NULL) {
        fprintf(stderr, "Error allocating memory for list_classes\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return 1;
    }
    type = H5Dget_type(dataset_id);
    native_type = h5tools_get_native_type(type);
    status = H5Dread(dataset_id, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, list_classes);
    if (status < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", DATASET_LIST_CLASSES);
    } else {
        printf("list_classes:\n");
        for (int i = 0; i < dims[0]; i++) {
            printf("%s\n", list_classes[i]);
        }
    }
    free(list_classes);
    H5Dclose(dataset_id);
    H5Tclose(datatype);
    H5Sclose(dataspace);

    // Read the train_set_x dataset
    dataset_id = H5Dopen(file_id, DATASET_TRAIN_SET_X, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", DATASET_TRAIN_SET_X);
        H5Fclose(file_id);
        return 1;
    }

    datatype = H5Dget_type(dataset_id);
    dataspace = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(dataspace);
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    printf("The dims are[0] = %ld  [1] = %ld [2] = %ld [3] = %ld \n",dims[0],  dims[1], dims[2], dims[3]);

    size_t train_set_x_size = dims[0] * dims[1] * dims[2] * dims[3];
    unsigned char *train_set_x = malloc(train_set_x_size * sizeof(unsigned char));  // Allocate memory

    //unsigned char (*train_set_x)[dims[1]][dims[2]][dims[3]] = malloc(dims[0] * sizeof(*train_set_x));  // Allocate memory
    //unsigned char train_set_x[209][64][64][3];  // Adjust dimensions accordingly
    
    if (train_set_x == NULL) {
        fprintf(stderr, "Error allocating memory for train_set_x\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return 1;
    }

    status = H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, train_set_x);
    if (status < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", DATASET_TRAIN_SET_X);
    } else {
       //printf("train_set_x[0][0][0]: %u\n", train_set_x[0][52][59][0]);  // Example output
       printf("train_set_x[0][0][0]: %u\n", train_set_x[52*59]);  // Example output
    }
    free(train_set_x);
    H5Dclose(dataset_id);
    H5Tclose(datatype);
    H5Sclose(dataspace);

    // Read the train_set_y dataset
    dataset_id = H5Dopen(file_id, DATASET_TRAIN_SET_Y, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", DATASET_TRAIN_SET_Y);
        H5Fclose(file_id);
        return 1;
    }

    datatype = H5Dget_type(dataset_id);
    dataspace = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_ndims(dataspace);
    H5Sget_simple_extent_dims(dataspace, dims, NULL);

    long long *train_set_y = malloc(dims[0] * sizeof(long long));  // Allocate memory
    //long long train_set_y[209];  // Assuming 64-bit integers
    //unsigned char train_set_x[200][64][64][3];  // Adjust dimensions accordingly
    if (train_set_y == NULL) {
        fprintf(stderr, "Error allocating memory for train_set_y\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return 1;
    }

    status = H5Dread(dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, train_set_y);
    if (status < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", DATASET_TRAIN_SET_Y);
    } else {
        printf("train_set_y:\n");
        for (int i = 0; i < dims[0]; i++) {
            printf("%lld ", train_set_y[i]);
        }
        printf("\n");
    }
    free(train_set_y);
    H5Dclose(dataset_id);
    H5Tclose(datatype);
    H5Sclose(dataspace);

    // Close the HDF5 file
    H5Fclose(file_id);

    return 0;
}
