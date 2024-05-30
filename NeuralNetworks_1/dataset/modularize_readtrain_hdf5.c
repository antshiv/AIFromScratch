#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

// Function prototypes
hid_t open_file(const char *file_name);
void close_file(hid_t file_id);
hid_t get_native_type(hid_t type);
void *read_dataset(hid_t file_id, const char *dataset_name, hid_t *datatype, hsize_t **dims, int *ndims);
void print_list_classes(char (*list_classes)[7], hsize_t size);
void print_train_set_x(unsigned char *train_set_x, hsize_t *dims);
void print_train_set_y(long long *train_set_y, hsize_t size);

int main() {
    // File and dataset names
    const char *file_name = "train_catvnoncat.h5";

    // Open the HDF5 file
    hid_t file_id = open_file(file_name);
    if (file_id < 0) return 1;

    // Read and print the list_classes dataset
    hid_t datatype;
    hsize_t *dims;
    int ndims;
    char (*list_classes)[7] = (char (*)[7])read_dataset(file_id, "list_classes", &datatype, &dims, &ndims);
    if (list_classes) {
        print_list_classes(list_classes, dims[0]);
        free(list_classes);
    }

    // Read and print the train_set_x dataset
    unsigned char *train_set_x = (unsigned char *)read_dataset(file_id, "train_set_x", &datatype, &dims, &ndims);
    if (train_set_x) {
        print_train_set_x(train_set_x, dims);
        free(train_set_x);
    }

    // Read and print the train_set_y dataset
    long long *train_set_y = (long long *)read_dataset(file_id, "train_set_y", &datatype, &dims, &ndims);
    if (train_set_y) {
        print_train_set_y(train_set_y, dims[0]);
        free(train_set_y);
    }

    // Cleanup
    free(dims);
    close_file(file_id);

    return 0;
}

hid_t open_file(const char *file_name) {
    hid_t file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error opening file: %s\n", file_name);
    }
    return file_id;
}

void close_file(hid_t file_id) {
    H5Fclose(file_id);
}

hid_t get_native_type(hid_t type) {
    H5T_class_t type_class = H5Tget_class(type);
    if (type_class == H5T_BITFIELD) {
        return H5Tcopy(type);
    } else {
        return H5Tget_native_type(type, H5T_DIR_DEFAULT);
    }
}

void *read_dataset(hid_t file_id, const char *dataset_name, hid_t *datatype, hsize_t **dims, int *ndims) {
    hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening dataset: %s\n", dataset_name);
        return NULL;
    }

    hid_t dataspace_id = H5Dget_space(dataset_id);
    *ndims = H5Sget_simple_extent_ndims(dataspace_id);
    *dims = (hsize_t *)malloc(*ndims * sizeof(hsize_t));
    if (*dims == NULL) {
        fprintf(stderr, "Error allocating memory for dimensions\n");
        H5Dclose(dataset_id);
        return NULL;
    }
    H5Sget_simple_extent_dims(dataspace_id, *dims, NULL);

    *datatype = H5Dget_type(dataset_id);
    hid_t native_type = get_native_type(*datatype);
    size_t size = H5Tget_size(native_type);
    for (int i = 0; i < *ndims; i++) {
        size *= (*dims)[i];
    }

    void *data = malloc(size);
    if (data == NULL) {
        fprintf(stderr, "Error allocating memory for data\n");
        H5Dclose(dataset_id);
        return NULL;
    }

    if (H5Dread(dataset_id, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        fprintf(stderr, "Error reading dataset: %s\n", dataset_name);
        free(data);
        data = NULL;
    }

    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Tclose(native_type);

    return data;
}

void print_list_classes(char (*list_classes)[7], hsize_t size) {
    printf("list_classes:\n");
    for (hsize_t i = 0; i < size; i++) {
        printf("%s\n", list_classes[i]);
    }
}

void print_train_set_x(unsigned char *train_set_x, hsize_t *dims) {
    printf("train_set_x[0][0][0]: %u\n", train_set_x[51 * dims[1] * dims[2] * dims[3]]);
}

void print_train_set_y(long long *train_set_y, hsize_t size) {
    printf("train_set_y:\n");
    for (hsize_t i = 0; i < size; i++) {
        printf("%lld ", train_set_y[i]);
    }
    printf("\n");
}
