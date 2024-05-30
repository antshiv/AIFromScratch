#include <stdio.h>
#include "hdf5.h"


int main() {
	hid_t file, dset, fspace;
	unsigned mode           = H5F_ACC_RDONLY;
	int ret_val;
	char     file_name[]    = "dataset/train_catvnoncat.h5";
	char     train_path[] = "train_set_x";
	char     train_path_y[] = "train_set_y";
	char     test_path[] = "test_set_x";
	H5O_info2_t info;
	int elts[100000];
	
	if ((file = H5Fopen(file_name, mode, H5P_DEFAULT)) == H5I_INVALID_HID) {
        	printf("Error loading H5 data \n");
		goto fail_file;
	}
	
	if ((dset = H5Dopen2(file, train_path, H5P_DEFAULT)) == H5I_INVALID_HID) {
		printf("Error in H5Open2 \n");
		goto fail_dset;
	}
       
	if (H5Oget_info_by_name(file, train_path, &info, H5O_INFO_BASIC | H5O_INFO_NUM_ATTRS, H5P_DEFAULT) < 0) {
		printf("Unable to get h5 info \n" ); 
		return 0;
	}

	// get the dataset's dataspace
        if ((fspace = H5Dget_space(dset)) == H5I_INVALID_HID) {
            	printf("Failed HSpace \n");
		goto fail_fspace;
        }

	const int ndims = H5Sget_simple_extent_ndims(fspace);
	printf("The NDims are %d \n", ndims); 
	
	//hsize_t dims[ndims];
	//H5Sget_simple_extent_dims(dspace, dims, NULL);


       // determine the object type
        switch (info.type) {
            case H5O_TYPE_GROUP:
                printf("HDF5 group\n");
                break;
            case H5O_TYPE_DATASET:
                printf("HDF5 dataset\n");
		if (H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, elts) < 0){
			printf("Error reading dataset");
			//return 0;
		}

		    
                break;
            case H5O_TYPE_NAMED_DATATYPE:
                printf("HDF5 datatype\n");
                break;
            default:
                printf("UFO?\n");
                break;
        }

	// print basic information
        printf("Reference count: %u\n", info.rc);
        printf("Attribute count: %lld\n", info.num_attrs);
 

	printf("H5 can be incldued \n");
	
	fail_update:
		H5Sclose(fspace);
	fail_fspace:
		H5Dclose(dset);
	fail_dset:
		H5Fclose(file);
	fail_file:;

	  return 0;
}


