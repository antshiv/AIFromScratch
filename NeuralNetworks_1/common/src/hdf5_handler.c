
#include <stdio.h>
#include <stdlib.h>
#include "hdf5_utils.h"
#include "nn.h"
#include <stdbool.h>

#define DATASET_TRAIN_SET_X "train_set_x"
#define DATASET_TRAIN_SET_Y "train_set_y"

void populate_dataset(dataset_t *dataset, const char *name, dataset_type_t type, const char *file_name)
{
    dataset->name = name;
    dataset->data = NULL;
    dataset->dims = NULL;
    dataset->ndims = 0;
    dataset->type = type;
    dataset->file_name = file_name;
}

int loaddata(dataset_t *dataset)
{
    printf("Loading dataset: %s\n", dataset->name);
    // Open the HDF5 files
    hid_t file_id = open_file(dataset->file_name);
    if (file_id < 0) {
        printf("Error opening file %s\n", dataset->file_name);
        return 1;
    }
    dataset->file_id = file_id;
    // Read the dataset
    hid_t datatype;
    hsize_t *dims;
    int ndims;
    void *data = read_dataset(file_id, dataset->name, &datatype, &dims, &ndims);
    if (!data)
    {
        close_file(file_id);
        return 1;
    }

    dataset->data = data;
    dataset->dims = dims;
    dataset->ndims = ndims;
    return 0;
}

void closedata(dataset_t *dataset)
{
    free(dataset->data);
    free(dataset->dims);
    close_file(dataset->file_id);
}