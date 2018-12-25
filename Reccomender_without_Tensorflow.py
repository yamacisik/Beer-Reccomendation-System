
import pandas as pd 
import numpy as np
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, find,lil_matrix
import tensorflow as tf




def calculate_loss(R,Q,P,train,test,z_i,z_j):
  
    hat=Q@P
    train_loss= np.mean(np.square(R[z_i[train],z_j[train]]-hat[z_i[train],z_j[train]]))
    val_loss=np.mean(np.square(R[z_i[test],z_j[test]]-hat[z_i[test],z_j[test]]))

    return train_loss,val_loss

def calc_accuracy(R,Q1,P1,train,test,z_i,z_j,change=0.5):
    r_hat=Q1@P1
    r_hat[r_hat>5]=5
    r_hat[r_hat<1]=1
    train_accuracy= np.sum(np.abs(r_hat[z_i[train],z_j[train]]-R[z_i[train],z_j[train]])<change)/len(train)
    test_accuracy= np.sum(np.abs(r_hat[z_i[test],z_j[test]]-R[z_i[test],z_j[test]])<change)/len(test)

    return train_accuracy,test_accuracy


def matrix_factor(X,train,validation,batch_size=1000,epochs=100,k=4,lr=0.009,beta=0.001):
    n,m=X.shape
    Q=np.random.rand(n,k)
    P=np.random.rand(m,k).T  
    #z_i,z_j,_=find(X)
    z_i,z_j=np.where(X>0)

    for epoch in range(epochs):
        if epoch%50==0:
            tlos,vlos=calculate_loss(X,Q,P,validation,train,z_i,z_j)
            print("Epoch : " + str(epoch)+ ", Training Loss: "+str(tlos)+", Validation Loss: "+str(vlos))
            ## Take batches
        batch=np.random.choice(train,size=batch_size,replace=False)     
        for u in batch:
            i=z_i[u]
            j=z_j[u]
            Q[i]=Q[i]+2*lr*(X[i,j]-Q[i]@P[:,j])*P[:,j]-beta*Q[i]
            P[:,j]=P[:,j]+2*lr*(X[i,j]-Q[i]@P[:,j])*Q[i]-beta*P[:,j]
    return Q,P
     

   Q1,P1=matrix_factor(R,train,validation,batch_size=400000,epochs=500,k=2,lr=0.005,beta=0.0001)