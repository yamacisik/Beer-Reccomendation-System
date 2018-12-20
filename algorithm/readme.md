This Folder contains the implementation of Matrix Factorization Algorithm. I use 3 different implementation for different performances:
* Vanilla Algorithm : Uses for loops and the formula of the gradient for each gradient descent step.
* Tensorflow for CPU : Basic Tensorflow implementation without any GPU support using the tensorflow optimizer.
* Tensorflow with GPU : Another tensorflow implementation where, the data is split further into batches to take advantage of parrallelization and not deal with OOM errors for GPU's
