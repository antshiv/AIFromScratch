
// Structure to hold the parameters W and b
typedef struct {
    double** W;
    double* b;
} LayerParameters_t;

// Structure to hold the result and cache
typedef struct {
    double* A;
    double* cache;
    int size;
} ActivationResult_t;


typedef struct layerdims {
    int* layer_dims;
    int L;
} layerdims_t;