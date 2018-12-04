import pandas as pd 
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from scipy import sparse
import pickle

df=pd.read_csv("beer_reviews.csv")
dfc=df[["review_overall","review_profilename","beer_name","beer_beerid","brewery_name"]]

## Function that creates the rating matrix and the neccesary dictionaries to interpret predictions

def rating_matrix(dfc):
    users=dfc.review_profilename
    beer_id=dfc.beer_beerid
    beer_name=dfc.beer_name
    brewery_name=dfc.brewery_name
    k=len(dfc.review_profilename.unique())
    j=len(dfc.beer_beerid.unique())
    ratings=dfc.review_overall
    R=np.zeros((k,j))
    
        ## Dictionary for Beer Id -Beer Name
    beer_into_name={}
    ## Dictionary for Index -Beer ID
    index_into_beerid={}
    ## Dictionary for Index -User ID
    index_to_userid={}

    u=0
    b=0

    for i in range(len(beer_id)):
        if beer_id[i] not in beer_into_name:
            beer_into_name[beer_id[i]]=str(beer_name[i]) +" by " + str(brewery_name[i])
            index_into_beerid[beer_id[i]]=u
            u+=1
        if users[i] not in index_to_userid:
            index_to_userid[users[i]]=b
            b+=1


    for i in range(len(ratings)):
        m=index_to_userid[users[i]]
        n=index_into_beerid[beer_id[i]]    
        R[m,n]=ratings[i]

    index_to_userid={v: k for k, v in index_to_userid.items()}    
    index_into_beerid={v: k for k, v in index_into_beerid.items()}
    
    return R,index_to_userid,index_into_beerid,beer_into_name

R,user_id,beer_id,beer_name=rating_matrix(dfc)


def matrix_factor(X,epochs=15000,k=4,lr=0.007,beta=0.001):
    n,m=X.shape
    Q=np.random.rand(n,k)
    P=np.random.rand(m,k).T  
    z_i,z_j=np.where(X>0)
    n=len(z_i)
    for k in range(epochs):
        print("Epoch : " + str(k))
        for u in range(n):
            i=z_i[u]
            j=z_j[u]
            Q[i]=Q[i]+2*lr*(X[i,j]-Q[i]@P[:,j])*P[:,j]-beta*Q[i]
            P[:,j]=P[:,j]+2*lr*(X[i,j]-Q[i]@P[:,j])*Q[i]-beta*P[:,j]
    return Q@P
     

    
r_hat=matrix_factor(R,k=4)

pickle.dump( r_hat, open( "matrix.p", "wb" ) )

pickle.dump( beer_id, open( "beer_id.p", "wb" ) )

pickle.dump( user_id, open( "user_id.p", "wb" ) )

pickle.dump( beer_name, open( "beer_name.p", "wb" ) )