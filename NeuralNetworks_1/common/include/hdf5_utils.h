#ifndef HDF5_UTILS_H
#define HDF5_UTILS_H

#include <hdf5.h>

typedef enum
{
    TRAIN,
    TEST
} dataset_type_t;

typedef struct dataset
{
    void *data;
    hsize_t *dims;
    int ndims;
    const char *name;
    dataset_type_t type;
    hid_t file_id;
    const char *file_name;
} dataset_t;

typedef struct {
    const char *name;
    dataset_type_t type;
    dataset_t *dataset;
    const char *file_name;
} dataset_info_t;

hid_t open_file(const char *file_name);
void close_file(hid_t file_id);
hid_t h5tools_get_native_type(hid_t type);
void *read_dataset(hid_t file_id, const char *dataset_name, hid_t *datatype, hsize_t **dims, int *ndims);
void get_rgb_values(const unsigned char *data, hsize_t height, hsize_t width, hsize_t channels,
                    hsize_t image_index, hsize_t pixel_row, hsize_t pixel_col,
                    unsigned char *r, unsigned char *g, unsigned char *b);

void populate_dataset(dataset_t *dataset, const char *name, dataset_type_t type, const char *file_name);
int loaddata(dataset_t *dataset);
void closedata(dataset_t *dataset);


#endif // HDF5_UTILS_H
