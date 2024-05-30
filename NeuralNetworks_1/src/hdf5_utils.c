#include "hdf5_utils.h"
#include <stdio.h>
#include <stdlib.h>

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

void get_rgb_values(const unsigned char *data, hsize_t height, hsize_t width, hsize_t channels,
                    hsize_t image_index, hsize_t pixel_row, hsize_t pixel_col,
                    unsigned char *r, unsigned char *g, unsigned char *b) {
    // Calculate the indices for the red, green, and blue channels
    size_t index_r = image_index * height * width * channels + pixel_row * width * channels + pixel_col * channels + 0;
    size_t index_g = image_index * height * width * channels + pixel_row * width * channels + pixel_col * channels + 1;
    size_t index_b = image_index * height * width * channels + pixel_row * width * channels + pixel_col * channels + 2;
    // Retrieve the RGB values
    *r = data[index_r];
    *g = data[index_g];
    *b = data[index_b];
}

