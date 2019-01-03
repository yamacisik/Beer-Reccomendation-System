import pandas as pd 
import numpy as np
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, find,lil_matrix
import tensorflow as tf
import pickle
import time



## Function that calculates both the training and test lost with given Q and P matrices
## Returns average training and test loss respectively
def calculate_loss(R,Q,P,train,test,z_i,z_j):
  
    r_hat=Q@P
    train_loss= np.mean(np.square(R[z_i[train],z_j[train]]-r_hat[z_i[train],z_j[train]]))
    test_loss=np.mean(np.square(R[z_i[test],z_j[test]]-r_hat[z_i[test],z_j[test]]))

    return train_loss,val_loss

## Function that calculates the accuracy of each calculated rating by an absolute difference. 
##@@ tolerance is the absolute difference parameter, higher tolerance higher accuracy

def calc_accuracy(R,Q,P,train,test,z_i,z_j,tolerance=0.25):
    r_hat=Q@P
    ## Since the ratings are supposed to be between 1 and 5, anything below 1 is set to 1
    ## and anything above 5 is set to 5
    r_hat[r_hat>5]=5
    r_hat[r_hat<1]=1
    train_accuracy= np.sum(np.abs(r_hat[z_i[train],z_j[train]]-R[z_i[train],z_j[train]])<change)/len(train)
    test_accuracy= np.sum(np.abs(r_hat[z_i[test],z_j[test]]-R[z_i[test],z_j[test]])<change)/len(test)

    return train_accuracy,test_accuracy


## Function that returns the resulting 2 matrices after the factorization. At each k'th report epoch
## the test and training loss are reported, returns two matrices Q,P where R_hat=QxP
## @lr-learning rate , @beta-regularization parameter, @epochs-number of iterations, @k-number of hidden latent factors 
## @report_epoch:frequency of training and test loss reports

def matrix_factor(X,train,test,epochs=10,k=4,lr=0.009,beta=0.001,report_epoch=2):
    n,m=X.shape
    Q=np.random.rand(n,k)
    P=np.random.rand(m,k).T  
    z_i,z_j=np.where(X>0)

    print("STARTING THE DESCENT")
    start=time.time()
    for epoch in range(epochs):
        if epoch%report_epoch==0:
            time_ellapsed=(np.round((time.time()-start)*1000)/1000)/report_epoch
            train_loss,test_loss=calculate_loss(X,Q,P,train,test,z_i,z_j)
            print("Epoch : " + str(epoch)+ ", Training Loss: "+str(train_loss)+", Test Loss: "+str(test_loss)+ ", Average time spent on each epoch: " + str(time_ellapsed))
            start=time.time()
        for u in range(len(z_i)):
            if u not in test:
                i=z_i[u]
                j=z_j[u]
                Q[i]=Q[i]+2*lr*(X[i,j]-Q[i]@P[:,j])*P[:,j]-beta*Q[i]
                P[:,j]=P[:,j]+2*lr*(X[i,j]-Q[i]@P[:,j])*Q[i]-beta*P[:,j]
    
    return Q,P
     
    