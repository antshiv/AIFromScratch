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
