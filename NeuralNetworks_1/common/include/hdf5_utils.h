#ifndef HDF5_UTILS_H
#define HDF5_UTILS_H

#include <hdf5.h>

hid_t open_file(const char *file_name);
void close_file(hid_t file_id);
hid_t h5tools_get_native_type(hid_t type);
void *read_dataset(hid_t file_id, const char *dataset_name, hid_t *datatype, hsize_t **dims, int *ndims);
void get_rgb_values(const unsigned char *data, hsize_t height, hsize_t width, hsize_t channels,
                    hsize_t image_index, hsize_t pixel_row, hsize_t pixel_col,
                    unsigned char *r, unsigned char *g, unsigned char *b);

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
    char *name;
    dataset_type_t type;
} dataset_t;

typedef struct {
    const char *name;
    dataset_type_t type;
    dataset_t dataset;
} dataset_info_t;



#endif // HDF5_UTILS_H
