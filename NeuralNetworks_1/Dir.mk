# Directories
SRC_DIR = src
INCLUDE_DIR = include
LOGISTIC_DIR = $(SRC_DIR)/logisticregression
BUILD_DIR = build

# Create build directory if it doesn't exist
$(shell mkdir -p $(BUILD_DIR))

# Source and object files
SRCS = $(wildcard $(SRC_DIR)/*.c) $(wildcard $(LOGISTIC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(patsubst $(LOGISTIC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS)))

# Target executable
TARGET = logisticregression

