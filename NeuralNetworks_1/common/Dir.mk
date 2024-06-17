# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj

# Create the object directory if it does not exist
$(shell mkdir -p $(OBJ_DIR))

# Compiler and flags
CC := gcc
CFLAGS := -I$(INC_DIR) -I../include -Wall -Wextra -Werror

# Files
SRC := $(wildcard $(SRC_DIR)/*.c)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET := libcommon.a
