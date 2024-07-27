#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "transformer.h"

void write_config_to_json(const char *file_path, TransformerConfig_t *config, transformer_parameters_t *params, size_t *param_sizes) {
    FILE *file = fopen(file_path, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        exit(1);
    }

    fprintf(file, "{\n");
    fprintf(file, "  \"config\": {\n");
    fprintf(file, "    \"num_layers\": %d,\n", config->num_layers);
    fprintf(file, "    \"embed_dim\": %d,\n", config->embed_dim);
    fprintf(file, "    \"batch_size\": %d,\n", config->batch_size);
    fprintf(file, "    \"num_heads\": %d,\n", config->num_heads);
    fprintf(file, "    \"feedforward_dim\": %d,\n", config->feedforward_dim);
    fprintf(file, "    \"dropout\": %.2f,\n", config->dropout);
    fprintf(file, "    \"vocab_size\": %d,\n", config->vocab_size);
    fprintf(file, "    \"max_length\": %d,\n", config->max_length);
    fprintf(file, "    \"num_classes\": %d\n", config->num_classes);
    fprintf(file, "  },\n");

    fprintf(file, "  \"parameters\": [\n");
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        fprintf(file, "    {\n");
        fprintf(file, "      \"size\": %zu,\n", param_sizes[i]);
        fprintf(file, "      \"pointer\": \"%p\"\n", (void *)(*(((float **)params) + i)));
        fprintf(file, "    }%s\n", (i == NUM_PARAMETER_TENSORS - 1) ? "" : ",");
    }
    fprintf(file, "  ]\n");
    fprintf(file, "}\n");

    fclose(file);
}
