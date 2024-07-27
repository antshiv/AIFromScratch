# Include the directory-specific makefile
include Dir.mk
BUILD_DIR = build

CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -L/home/antshiv/Programs/lib -lhdf5 -lm
LDFLAGS= 

# Default Rule
all: $(BUILD_DIR)/$(TARGET)

debug: CFLAGS += -DDEBUG -g
debug: $(BUILD_DIR)/$(TARGET)
	@touch debug_build

# Release rule (optional, to explicitly build without debug info)
release: CFLAGS += -O2
release: $(BUILD_DIR)/$(TARGET)

# Link object files into executable
$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS) 

# Compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(LOGISTIC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all debug release clean
