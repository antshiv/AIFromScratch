#ifndef LOSS_H
#define LOSS_H

// Function prototypes for loss calculations
double euclidean_distance(double* vec1, double* vec2, int length);
double contrastive_loss_single(double* anchor, double* positive_or_negative, int label, int embedding_length, double margin);
double contrastive_loss_batch(double** embeddings, int* labels, int num_pairs, int embedding_length, double margin);
double triplet_loss_single(double* anchor, double* positive, double* negative, int embedding_length, double margin);
double triplet_loss_batch(double** embeddings, int num_triplets, int embedding_length, double margin);

#endif // LOSS_H
