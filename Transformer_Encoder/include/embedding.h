#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "attention.h"

Matrix_t init_random_embeddings(int vocab_size, int embedding_dim);
Matrix_t init_xavier_embeddings(int vocab_size, int embedding_dim);

#endif // EMBEDDING_H
