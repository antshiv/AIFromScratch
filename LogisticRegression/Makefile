CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -L/home/antshiv/Programs/lib -lhdf5
LDFLAGS= 


# Source and target files
SRC_DIR = src
INCLUDE_DIR = include
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=%.o) 
TARGET = read_hdf5

all: $(TARGET)

debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)
	@touch debug_build

# Release rule (optional, to explicitly build without debug info)
release: CFLAGS += -O2
release: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS) 

# Compile source files into object files
%.o: $(SRC_DIR)/%.c
	$(CC) -c $< -o $@ $(CFLAGS) 


# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET) read_hdf5


.PHONY: all debug release clean