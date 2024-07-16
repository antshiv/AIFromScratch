#ifndef LOSS_BACKWARD_H
#define LOSS_BACKWARD_H

// Function prototypes for backpropagation of loss
void contrastive_loss_backward(double* grad_anchor, double* grad_positive_or_negative, double* anchor, double* positive_or_negative, int label, int embedding_length, double margin);
void triplet_loss_backward(double* grad_anchor, double* grad_positive, double* grad_negative, double* anchor, double* positive, double* negative, int embedding_length, double margin);

#endif // LOSS_BACKWARD_H
