#include <math.h>
#include "loss_backward.h"
#include "loss.h"

void contrastive_loss_backward(double* grad_anchor, double* grad_positive_or_negative, double* anchor, double* positive_or_negative, int label, int embedding_length, double margin) {
    double dist = euclidean_distance(anchor, positive_or_negative, embedding_length);
    double factor = (label == 1) ? 1.0 : (dist < margin) ? -1.0 : 0.0;

    for (int i = 0; i < embedding_length; i++) {
        grad_anchor[i] = factor * (anchor[i] - positive_or_negative[i]) / (dist + 1e-8);
        grad_positive_or_negative[i] = factor * (positive_or_negative[i] - anchor[i]) / (dist + 1e-8);
    }
}

void triplet_loss_backward(double* grad_anchor, double* grad_positive, double* grad_negative, double* anchor, double* positive, double* negative, int embedding_length, double margin) {
    double pos_dist = euclidean_distance(anchor, positive, embedding_length);
    double neg_dist = euclidean_distance(anchor, negative, embedding_length);
    double factor_pos = (pos_dist - neg_dist + margin > 0) ? 1.0 : 0.0;
    double factor_neg = (pos_dist - neg_dist + margin > 0) ? -1.0 : 0.0;

    for (int i = 0; i < embedding_length; i++) {
        grad_anchor[i] = (factor_pos * (anchor[i] - positive[i]) / (pos_dist + 1e-8)) + (factor_neg * (anchor[i] - negative[i]) / (neg_dist + 1e-8));
        grad_positive[i] = factor_pos * (positive[i] - anchor[i]) / (pos_dist + 1e-8);
        grad_negative[i] = factor_neg * (negative[i] - anchor[i]) / (neg_dist + 1e-8);
    }
}
