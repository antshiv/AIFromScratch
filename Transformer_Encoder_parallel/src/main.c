#include <stdio.h>
#include "transformer.h"


int main() {
    printf("Hello, World!\n");
    int num_layers = 6;
    int embed_dim = 384;
    int num_heads = 8;
    int feedforward_dim = 2048;
    int batch_size = 64;
    int vocab_size = 32000;
    int max_length = 256;
    int num_classes = NER_VOCAB_SIZE;

    // input X data
    // output Y data
    // token/ vocabulary data
    // embedding data


    train_transformer(num_layers, embed_dim, num_heads, feedforward_dim, batch_size, vocab_size, max_length, num_classes);
    return 0;
}
