# Beer-Reccomendation-System

This repository contains a Beer-Reccomendation-System that uses scraped data from Beer-Advocate and a matrix factorization algorithm. I compare the results of 3 different implementation, a simple implementation that uses for loops and 2 tensorflow aplications with and without GPU support. 

* Vanilla Algorithm : Uses for loops and the formula of the gradient for each gradient descent step.
* Tensorflow for CPU : Basic Tensorflow implementation without any GPU support using the tensorflow optimizer.
* Tensorflow with GPU : Another tensorflow implementation where the data is split further into batches to take advantage of parrallelization and avoid OOM errors for GPU's

For both tensorflow implementations I use version 1.12 to take advantage of unsafe divide function tf.div_no_nan 


### Preprocessing
Before starting to train, we need preprocess the data. In order to have more accurate predictions I dropped users and beers less then 50 reviews. Then the data was split test and training sets with a 1/9 ratio. The final training set has 12329 users and 11002 beers.

