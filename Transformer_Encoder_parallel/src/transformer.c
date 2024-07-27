#include <stdio.h>
#include <stdlib.h>
#include "transformer.h"
#include "computational_graph.h"

ner_vocabulary_t ner_vocabulary_dict[] = {
    {"O", 0},
    {"B-SECTION", 1},
    {"I-SECTION", 2},
    {"SECTION-NUM", 3},
    {"B-SUBSECTION", 4},
    {"I-SUBSECTION", 5},
    {"SUBSECTION-NUM", 6},
    {"B-ACT", 7},
    {"I-ACT", 8},
    {"ACT-NUM", 9},
    {"B-REG", 10},
    {"I-REG", 11},
    {"REG-NUM", 12},
};

/* based on the transformer config lets us allocate memory
 * for the parameters
 */

TransformerConfig_t init_parameters_config(int num_layers,
                                           int embed_dim,
                                           int num_heads,
                                           int feedforward_dim,
                                           int batch_size,
                                           int vocab_size,
                                           int max_length,
                                           int num_classes)
{
    TransformerConfig_t config;
    config.num_layers = num_layers;
    config.embed_dim = embed_dim;
    config.num_heads = num_heads;
    config.feedforward_dim = feedforward_dim;
    config.batch_size = batch_size;
    config.vocab_size = vocab_size;
    config.max_length = max_length;
    config.num_classes = num_classes;
    return config;
}

void print_config(TransformerConfig_t config)
{
    printf("Number of layers: %d\n", config.num_layers);
    printf("Embedding dimension: %d\n", config.embed_dim);
    printf("Number of heads: %d\n", config.num_heads);
    printf("Feedforward dimension: %d\n", config.feedforward_dim);
    printf("Batch size: %d\n", config.batch_size);
    printf("Vocabulary size: %d\n", config.vocab_size);
    printf("Maximum length: %d\n", config.max_length);
    printf("Number of classes: %d\n", config.num_classes);
}

void print_ner_vocabulary()
{
    printf("NER Vocabulary\n");
    for (int i = 0; i < NER_VOCAB_SIZE; i++)
    {
        printf("Tag: %s, ID: %d\n", ner_vocabulary_dict[i].tag, ner_vocabulary_dict[i].id);
    }
}

void fill_in_parameter_size(size_t *param_sizes, TransformerConfig_t config)
{
    size_t V = config.vocab_size;
    size_t embed_dim = config.embed_dim;
    size_t num_heads = config.num_heads;
    size_t feedforward_dim = config.feedforward_dim;
    size_t max_len = config.max_length;
    size_t L = config.num_layers;
    size_t num_classes = config.num_classes;
    param_sizes[0] = V * embed_dim;                             // tk
    param_sizes[1] = max_len * embed_dim;                       // pos_enc
    param_sizes[2] = L * embed_dim;                             // ln1w
    param_sizes[3] = L * embed_dim;                             // ln1b
    param_sizes[4] = L * embed_dim * 3 * embed_dim / num_heads; // qkvw
    param_sizes[5] = L * embed_dim / num_heads;                 // qkvb
    param_sizes[6] = L * embed_dim * embed_dim;                 // attnprojw
    param_sizes[7] = L * embed_dim;                             // attnprojb
    param_sizes[8] = L * embed_dim;                             // ln2w
    param_sizes[9] = L * embed_dim;                             // ln2b
    param_sizes[10] = L * embed_dim * feedforward_dim;          // ff1w
    param_sizes[11] = L * feedforward_dim;                      // ff1b
    param_sizes[12] = L * feedforward_dim * embed_dim;          // ff2w
    param_sizes[13] = L * embed_dim;                            // ff2b
    param_sizes[14] = L * embed_dim;                            // ln3w
    param_sizes[15] = L * embed_dim;                            // ln3b
    param_sizes[16] = L * num_classes * embed_dim;              // nerw
    param_sizes[17] = L * num_classes;                          // nerb
}

float *malloc_and_point_parameters(transformer_parameters_t *params, size_t *param_sizes)
{
    // allocate all the parameters in one malloc to keep all the
    // parameters in one contiguous block of memory
    size_t total_size = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        total_size += param_sizes[i];
    }

    float *params_mem = (float *)malloc(total_size * sizeof(float));
    if (params_mem == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    // point the parameters to the correct location in the memory
    size_t offset = 0;
    float **param_offsets[] = {
        &params->tk,
        &params->pos_enc,
        &params->ln1w,
        &params->ln1b,
        &params->qkvw,
        &params->qkvb,
        &params->attnprojw,
        &params->attnprojb,
        &params->ln2w,
        &params->ln2b,
        &params->ff1w,
        &params->ff1b,
        &params->ff2w,
        &params->ff2b,
        &params->ln3w,
        &params->ln3b,
        &params->nerw,
        &params->nerb};

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        *param_offsets[i] = params_mem + offset;
        offset += param_sizes[i];
    }
    return params_mem;
}

void free_parameters(float *params)
{
    free(params);
}

void cross_entropy_loss(float *y, float *y_hat, int batch_size, size_t num_classes)
{
    float loss = 0;
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < num_classes; j++)
        {
            loss += y[i * num_classes + j] * log(y_hat[i * num_classes + j]);
        }
    }
    loss = -loss / batch_size;
    printf("Cross entropy loss: %f\n", loss);
}

void train_transformer(int num_layers,
                       int embed_dim,
                       int num_heads,
                       int feedforward_dim,
                       int batch_size,
                       int vocab_size,
                       int max_length,
                       int num_classes)
{
    TransformerConfig_t config = init_parameters_config(num_layers, embed_dim, num_heads, feedforward_dim, batch_size, vocab_size, max_length, num_classes);
    print_config(config);
    print_ner_vocabulary();
    size_t param_sizes[NUM_PARAMETER_TENSORS];

    fill_in_parameter_size(param_sizes, config);

    transformer_parameters_t params;
    float *params_memory = malloc_and_point_parameters(&params, param_sizes);

    // Start initializing the parameters
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        printf("Parameter %d size: %zu\n", i, param_sizes[i]);
        for (int j = 0; j < param_sizes[i]; j++)
        {
            params.tk[j] = (float)rand() / RAND_MAX;
        }
    }

    // Start the forward pass
    /*
        Step one is to get the input data
        step two is to get the token from the data
        step three is to assiciate a random embedding to the token
        step four is to add the positional encoding to the embedding
        step 5 is to add the layernornalization of the embedding
        step 6 is to multiple this positional encoding with Q, K, V
        step 7 is to multiply Q*K / num_heads
        step 8 is to apply softmax to the output of Q*K
        step 9 is to multiply the output of softmax with V
        step 10 is to multiply the output of step 9 with W_O
        step 11 is to add the output of step 5 to this output
        step 12 is to apply layer normalization to the output of step 11
        step 13 is to apply feedforward to the output of step 12
        step 14 is to pass this to a higher dimenion feedforward layer
        step 15 is to project the 2nd layer of the feedfoard to the same dimension of the embeddings
        step 16 is to add the output of step 15 to the output of step 12
        step 17 is to apply layer normalization to the output of step 16
        step 18 to repeat the steps 5 to 17 for the number of layers
        step 19 is to pass the output of the last layer to the NER layer
    */
    // Calculate the loss or cross entropy and then we do the backward pass
    /*
         step 1 is to calculate the loss
         step 2 calculate the gradient of the NER w+b with respect to the loss
         Step 3 calculate the linear layernorn (w+b) with respect to the loss
         step 4 calculate the gradient of the feedforward layer 2 with respect to the loss
         step 5 calculate the gradient of the feedforward layer 1 with respect to the loss
         step 6 calculate the gradient of the layer normalization 2 (attn head output) with respect to the loss
         step 7 calculate the gradient of the W_O (W+b) with respect to the loss
         step 8 calculate the gradient of the V (dim/num_heads) with respect to the loss
         step 9 calculate the gradient of the Q*K with respect to the loss
         step 10 calculate the gradients of layernorm 1 (P.E) with respect to the loss
         step 11 repeat the steps 3 to 10 for the number of layers
         step 12 calculate the gradient of the embeddings with respect to the loss
    */

    /*
    Forward Pass:

      Get the input data.
      Get the token from the data.
      Associate a random embedding to the token.
      Add the positional encoding to the embedding.
      Apply layer normalization to the embedding.
      Multiply the normalized embedding with Q, K, V.
      Compute Q*K and divide by the square root of the embedding dimension.
      Apply softmax to the output of Q*K.
      Multiply the softmax output with V.
      Multiply the output of step 9 with W_O.
      Add the output of step 5 to the output of step 10.
      Apply layer normalization to the output of step 11.
      Apply the first feedforward layer to the output of step 12.
      Pass the output to a higher dimension feedforward layer.
      Project the second feedforward layer output to the embedding dimension.
      Add the output of step 15 to the output of step 12.
      Apply layer normalization to the output of step 16.
      Repeat steps 5 to 17 for the number of layers.
      Pass the output of the last layer to the NER layer.

  Backward Pass:

      Calculate the loss.
      Calculate the gradient of the NER weights and biases with respect to the loss.
      Calculate the gradient of the final layer normalization weights and biases with respect to the loss.
      Calculate the gradient of the second feedforward layer with respect to the loss.
      Calculate the gradient of the first feedforward layer with respect to the loss.
      Calculate the gradient of the second layer normalization (attention head output) with respect to the loss.
      Calculate the gradient of W_O (weights and biases) with respect to the loss.
      Calculate the gradient of V (dimension/num_heads) with respect to the loss.
      Calculate the gradient of Q*K with respect to the loss.
      Calculate the gradients of the first layer normalization (P.E.) with respect to the loss.
      Repeat steps 3 to 10 for the number of layers.
      Calculate the gradient of the embeddings with respect to the loss.
    */

   /* LEts calculate the dimension of each activation function with minibatch and layers */
   /*
    P.E = (batch_size, max_len, embed_dim)
    LN1 = (layer , batch_size, max_len, embed_dim)
    QKV = (layer, batch_size, max_len, embed_dim)
    attnProj = (layer, batch_size, max_len, embed_dim)
    residual1 = (layer, batch_size, max_len, embed_dim)
    LN2_mean = (layer, batch_size, max_len, embed_dim)
    LN2_rtsd = (layer, batch_size, max_len, embed_dim)
    FF1 = (layer, batch_size, feedforward_Dimension,  max_len, embed_dim)
    FF2 = (layer, batch_size, max_len, embed_dim, feedforward_Dimension)
    residual2 = (layer, batch_size, max_len, embed_dim)
    LN3_mean = (layer, batch_size, max_len, embed_dim)
    LN3_rtsd = (layer, batch_size, max_len, embed_dim)
    NER = (layer, batch_size, max_len, num_classes)
   */

   // Step 1 is to get the input data

    int y;
    int y_hat;

    const char *file_path = "transformer_config.json";
    write_config_to_json(file_path, &config, &params, param_sizes);

    printf("Configuration written to %s\n", file_path);    


    free_parameters(params_memory);
}