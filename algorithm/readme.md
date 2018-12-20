This Folder contains the implementation of Matrix Factorization Algorithm. I use 3 different implementation for different performances:
* Vanilla Algorithm : Uses for loops and the formula of the gradient for each gradient descent step.
* Tensorflow for CPU : Basic Tensorflow implementation without any GPU support using the tensorflow optimizer.
* Tensorflow with GPU : Another tensorflow implementation where the data is split further into batches to take advantage of parrallelization and avoid OOM errors for GPU's

For both tensorflow implementations I use version 1.12 to take advantage of unsafe divide function tf.div_no_nan 


# Preprocessing
Before starting to train, we need preprocess the data. In order to have more accurate predictions I dropped users and beers less then 50 reviews. Then the data was split test and training sets with a 1/9 ratio.
