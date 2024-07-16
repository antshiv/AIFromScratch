#include "transformer.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdalign.h> // For alignment macros
#include <string.h>

// Function to display help information
void display_help(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  -h                Display this help message and exit\n");
    printf("  -l <num_layers>   Number of transformer layers (default: 12)\n");
    printf("  -n <num_heads>    Number of attention heads (default: 12)\n");
    printf("  -s <hidden_size>  Hidden size / Embedding dim (default: 768)\n");
    printf("  -t <seq_length>   Sequence length (default: 512)\n");
    printf("  -b <batch_size>   Mini-batch size (default: 32)\n");
    printf("  -a <alignment>    Alignment for memory (default: 64)\n");
    printf("  -i <input_file>   Path to input data file\n");
}

int main(int argc, char **argv) {
        // Default values for the parameters
    int num_layers = 12;
    int num_heads = 12;
    int hidden_size = 768;
    int seq_length = 512;
    int batch_size = 32;
    int alignment = 64;
    char *input_file = NULL;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0) {
            display_help(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            num_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            hidden_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            seq_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            alignment = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else {
            fprintf(stderr, "Unknown option or missing argument: %s\n", argv[i]);
            display_help(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

        // Check if input_file is specified
    if (input_file == NULL) {
        fprintf(stderr, "Input file is required. Use -i <input_file> to specify the input data file.\n");
        display_help(argv[0]);
        exit(EXIT_FAILURE);
    }

    // Print the parsed parameters for verification
    printf("Parameters:\n");
    printf("  Number of layers: %d\n", num_layers);
    printf("  Number of heads: %d\n", num_heads);
    printf("  Hidden size: %d\n", hidden_size);
    printf("  Sequence length: %d\n", seq_length);
    printf("  Batch size: %d\n", batch_size);
    printf("  Alignment: %d\n", alignment);


    // Call the transformer model
    Transformer_t transformer = init_transformer(num_layers, num_heads, hidden_size, batch_size, alignment);
    printf("Transformer model initialized\n");


    return 0;
}
