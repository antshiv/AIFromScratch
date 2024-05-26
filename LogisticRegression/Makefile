# Include the directory-specific makefile
include Dir.mk

CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -L/home/antshiv/Programs/lib -lhdf5 -lm
LDFLAGS= 

#Default Rule
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
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(LOGISTIC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ 

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET) read_hdf5


.PHONY: all debug release clean