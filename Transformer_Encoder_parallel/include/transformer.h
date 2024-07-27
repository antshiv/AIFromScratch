#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "utils.h"

typedef struct ner_vocabulary {
    const char *tag;
    int id;
} ner_vocabulary_t;

// Declare the array as extern
#define NER_VOCAB_SIZE 13
extern ner_vocabulary_t ner_vocabulary_dict[NER_VOCAB_SIZE];

typedef struct TransformerConfig {
    int num_layers; // Number of Transformer layers
    int embed_dim; // Dimension of the embeddings
    int batch_size; // Batch size
    int num_heads; // Number of attention heads
    int feedforward_dim; // Dimension of the feedforward layer
    float dropout; // Dropout probability
    int vocab_size; // Size of the vocabulary
    int max_length; // Maximum length of the input sequence
    int num_classes; // Number of classes for classification
} TransformerConfig_t;

#define NUM_PARAMETER_TENSORS 18
typedef struct transformer_arameters {
    float *tk; //(V, embed_Dim)
    float *pos_enc; //(max_length, embed_dim)
    float ln1w; //(embed_dim)
    float ln1b; //(embed_dim)
    float *qkvw; //(embed_dim/atten_head, embed_dim/atten_head)*3
    float *qkvb; //(embed_dim/atten_head)
    float *attnprojw; //(embed_dim, embed_dim[concantenated])
    float *attnprojb; //(embed_dim)
    float ln2w; //(embed_dim)
    float ln2b; //(embed_dim)
    float ff1w; //(embed_dim, feedforward_dim)
    float ff1b; //(feedforward_dim)
    float ff2w; //(feedforward_dim, embed_dim)
    float ff2b; //(embed_dim)
    float ln3w; //(embed_dim)
    float ln3b; //(embed_dim)
    float nerw; //(num_classes, embed_dim)
    float nerb; //(num_classes)
} transformer_parameters_t;

#define NUM_ACTIVATION_TENSORS 6
typedef struct {
    float *pe_encode; //(batch_size, max_length, embed_dim)
    float *ln1; //(layers, batch_size, max_length, embed_dim)
    float *qkv; //(layers, batch_size, 3*embed_dim)
    float *attnproj; //(batch_size, max_length, embed_dim)
    float ln2; //(batch_size, max_length, embed_dim)
    float *ff1; //(batch_size, feedforward_dim, embed_dim)

} ActivationTensor_t;





typedef struct {
    TransformerConfig_t config;
    // the weights (parameters) of the model, and their sizes
    transformer_parameters_t params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    transformer_parameters_t grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensor_t acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensor_t grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} TransformerModel;

typedef struct TransformerLayers {
    float *input; // (batch_size, max_length, embed_dim)
    float *pe; // Positional encoding (batch_size, max_length, embed_dim)
    float *qa; // Queries activation (batch_size, max_length, embed_dim/num_heads)
    float *ka; // Keys activation (batch_size, max_length, embed_dim/num_heads)
    float *va; // Values activation (batch_size, max_length, embed_dim/num_heads)
    float *softmax; // Softmax output (batch_size, max_length, qk)
    float *qkv_head; // Perr Head queries/keys/values (batch_size, max_length, embed_dim/num_heads)
    float *qkv; // Concatenated queries/keys/values (batch_size, max_length, 3 * embed_dim)
    float *qkv_residual; // Residual 1 (batch_size, max_length, embed_dim)
    float *norm1; // Layer norm 1 - Attention output (batch_size, max_length, embed_dim)
    float *l1nna; // Layer 1 feed-forward network activation (batch_size, feedforward_dim, max_length, embed_dim)
    float *norm2; // Layer norm 2 - Feedforward output (batch_size, max_length, embed_dim)
    float *ner_act; // NER classification activation (batch_size, max_length, num_classes)
    float *loss; // Loss (batch_size, max_length)
} TransformerLayers_t;

void train_transformer(int num_layers,
                       int embed_dim,
                       int num_heads,
                       int feedforward_dim,
                       int batch_size,
                       int vocab_size,
                       int max_length,
                       int num_classes);

#endif