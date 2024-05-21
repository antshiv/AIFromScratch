CC = gcc
CFLAGS = -I./include -L/home/antshiv/Programs/lib -lhdf5

SRC = src/main.c src/hdf5_utils.c 
OBJ = $(SRC:.c=.o)

all: read_hdf5

read_hdf5: $(OBJ)
	$(CC)  -o $@ $^ $(CFLAGS)

clean:
	rm -f read_hdf5 $(OBJ)