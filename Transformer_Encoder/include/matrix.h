typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix_t;

Matrix_t init_matrix(int rows, int cols);
void free_matrix(Matrix_t mat);
Matrix_t matmul(const Matrix_t *A, const Matrix_t *B);
Matrix_t transpose(const Matrix_t *A);