# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(TransformerProject)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Add include directory
include_directories(include)
include_directories(/home/antshiv/Programs/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add source files
set(SOURCES
    src/main.c
 #   src/attention.c
 #   src/attention_backward.c
 #   src/feedforward.c
 #   src/feedforward_backward.c
 #   src/positional_encoding.c
     src/transformer.c
 #   src/transformer_backward.c
     src/utils.c
 #   src/loss.c
 #   src/loss_backward.c
)

# Add TensorFlow and Protobuf library paths
link_directories(/home/antshiv/Programs/lib)

# Create executable
add_executable(transformer_encoder ${SOURCES})

# Link libraries
target_link_libraries(transformer_encoder
    m
    tensorflow
)

# add linkers for the math library
target_link_libraries (transformer_encoder m)