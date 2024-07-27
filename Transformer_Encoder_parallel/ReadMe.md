# Encoder Transformer model for parallel processing

There are three main inputs to the model:
- The batch size or the number of sentences or input data in a batch   
- The number of tokens in each sentence or input data (256 or 512)
- The number of features in each token (768 or 1024) a.k.a the embedding dimension

The repo is designed to process all the input at once using parallel computing.

If we had to draw a 3D matrix of the input data, it would look like this:
- The first dimension is the batch size
- The second dimension is the number of tokens in each sentence
- The third dimension is the number of features in each token