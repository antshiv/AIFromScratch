#include <math.h>
#include "loss.h"

double euclidean_distance(double* vec1, double* vec2, int length) {
    double distance = 0.0;
    for (int i = 0; i < length; i++) {
        distance += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return sqrt(distance);
}

double contrastive_loss_single(double* anchor, double* positive_or_negative, int label, int embedding_length, double margin) {
    double dist = euclidean_distance(anchor, positive_or_negative, embedding_length);
    if (label == 1) {
        // Similar pair
        return dist;
    } else {
        // Dissimilar pair
        return fmax(0, margin - dist);
    }
}

double contrastive_loss_batch(double** embeddings, int* labels, int num_pairs, int embedding_length, double margin) {
    double total_loss = 0.0;
    for (int i = 0; i < num_pairs; i++) {
        double* anchor = embeddings[i * 2];
        double* positive_or_negative = embeddings[i * 2 + 1];
        total_loss += contrastive_loss_single(anchor, positive_or_negative, labels[i], embedding_length, margin);
    }
    return total_loss / num_pairs;
}

double triplet_loss_single(double* anchor, double* positive, double* negative, int embedding_length, double margin) {
    double pos_dist = euclidean_distance(anchor, positive, embedding_length);
    double neg_dist = euclidean_distance(anchor, negative, embedding_length);
    return fmax(0, pos_dist - neg_dist + margin);
}

double triplet_loss_batch(double** embeddings, int num_triplets, int embedding_length, double margin) {
    double total_loss = 0.0;
    for (int i = 0; i < num_triplets; i++) {
        double* anchor = embeddings[i * 3];
        double* positive = embeddings[i * 3 + 1];
        double* negative = embeddings[i * 3 + 2];
        total_loss += triplet_loss_single(anchor, positive, negative, embedding_length, margin);
    }
    return total_loss / num_triplets;
}
