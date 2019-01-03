import pandas as pd 
import numpy as np
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, find,lil_matrix
import tensorflow as tf
import pickle
import time
from reccomender_without_tensorflow import calc_accuracy,add_user



## Function for matrix factorization using tensoflow without GPU support, 
## Takes main matrix to be filled and the indices of training and test sets
## @lr-learning rate , @beta-regularization parameter, @epochs-number of iterations, @k-number of hidden latent factors 
## @report_epoch:frequency of training and test loss reports, @tolerance-threshold of the loss that stops the iteration process
## @early_stop_rate-the number of consecetive increase of the training losses required for stopping the iteration, if 0 no early stop

def matrix_factorization_tensorflow_cpu(X,train,test,k=3,lr=0.01,beta=0.00005,epochs=25000,early_stop_rate=1,report_epoch=10):
    m,n=X.shape
    z_i,z_j=np.where(X>0)
    
    ## Test and Training Set, I use two different matrices X_test and X_train to take advantage of the div_no_nan function

    X_train=X.copy()
    X_test=X.copy()
    X_test[z_i[train],z_j[train]]=0
    X_train[z_i[test],z_j[test]]=0
    
    Rating= tf.placeholder(tf.float32, [ m,n])
    Q = tf.Variable(tf.random.normal([m, k], stddev=0.2, mean=0), name="users")
    P=tf.Variable(tf.random.normal([k, n], stddev=0.2, mean=0), name="beers") 
    
    
    ## Slicing the indexing r_hat matrix creates variety of problems such as unefficient use of memory,
    ## with Tensorflow 1.12 we can use div_no_nan function by multiplying the r_hat by our initial Rating matrix and 
    ## dividing it again using div_no_nan method. This would make the values that were not initially rated to be 0 
    ## It's importonta to divid the total squared difference by the number of ratings in the training set rather than using
    ## mean function
    
    r_hat=tf.matmul(Q,P)
    loss= tf.div_no_nan((r_hat*Rating),Rating) ## Multiply and divide by Rating
    reg=beta*(tf.norm(Q)+tf.norm(P)) ## Regularization part of the loss
    loss = tf.reduce_sum(tf.squared_difference(Rating,loss))/len(train) +reg
    
  
    ## Optimizer, we use gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)
    
    config = tf.ConfigProto(device_count = {'GPU': 0}) ## Non GPU configuration
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    
    
    print("STARTING THE DESCENT")
    start=time.time()
    losses=[]
    for i in range(epochs):
        if i % report_epoch == 0:
            time_ellapsed=(np.round((time.time()-start)*1000)/1000)/report_epoch
            train_loss=sess.run(loss,feed_dict={Rating:X_train})
            test_loss=len(train)*sess.run(loss,feed_dict={Rating:X_test})/len(test)
            print("EPOCH " +str(i)+ ", Training Loss: " + str(train_loss)+ ", Test Loss: " + str(test_loss)+", Average time spent on each epoch: " + str(time_ellapsed))
            
            ## Mechanism for early stopping
            if early_stop_rate>0 and len(losses)>=early_stop_rate:
                count=0
                cur=losses[-1]
                for i in range(early_stop_rate):
                    if losses[-(i+2)]<losses[-(i+1)]:
                        count+=1
                if count==eary_stop_rate:
                    break
                    
            start=time.time()
        sess.run(train_step,feed_dict={Rating:X_test})
    
    q_f=sess.run(Q)
    p_f=sess.run(P)
    sess.close()
    return q_f,p_f
    