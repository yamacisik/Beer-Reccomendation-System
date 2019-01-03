import pandas as pd 
import numpy as np
import tensorflow as tf

df=pd.read_csv("final_data.csv")
dfc=df[["score_overall","user_id","beer_names","beer_id","brewery_name"]].dropna()

##df=pd.read_csv("beer_reviews.csv")
##dfc=df[["review_overall","review_profilename","beer_name","beer_beerid","brewery_name"]].dropna()

beers=dfc.groupby('beer_id').count().query("score_overall >=50").index
users=dfc.groupby('user_id').count().query("score_overall >=50").index
df_filtered=dfc[dfc.beer_id.isin(beers)][dfc.user_id.isin(users)]

users=pd.factorize(df_filtered.user_id)[0]
beers=pd.factorize(df_filtered.beer_id)[0]
index_to_userid=dict(zip(users,df_filtered.user_id))
index_into_beerid=dict(zip(beers,df_filtered.beer_id))

index_into_beerid = {v: k for k, v in index_into_beerid.items()}
index_to_userid = {v: k for k, v in index_to_userid.items()}
##R=[]
R=np.zeros((len(index_into_beerid),len(index_to_userid))).T

for index, row in df_filtered.iterrows():
    R[index_to_userid[row['user_id']],index_into_beerid[row['beer_id']]]=row['score_overall']
    ##R.append((index_to_userid[row['user_id']],index_into_beerid[row['beer_id']],row['score_overall']))
    
index_into_beerid = {v: k for k, v in index_into_beerid.items()}
index_to_userid = {v: k for k, v in index_to_userid.items()}

index_into_beername=dict(zip(beers, df_filtered.beer_names+ " by "+ df_filtered.brewery_name ))




X=np.array([[1,0,0,0,5],[0,0,4,1,0],[0,2,1,0,0],[0,0,4,4,1],[2,3,1,0,0],[5,3,1,0,0]],dtype="float64")

def Matrix_Factorization(X,k=3,lr=0.001,beta=0.0005,epochs=25000,tolerance=0.001):
    m,n=X.shape
    Rating= tf.placeholder(tf.float32, [ m,n])
    Q = tf.Variable(tf.truncated_normal([m, k], stddev=0.2, mean=0), name="users")
    P=tf.Variable(tf.truncated_normal([k, n], stddev=0.2, mean=0), name="beers") 
    
    r_hat=tf.matmul(Q,P)
    loss1= tf.div_no_nan((r_hat*Rating),Rating)
    loss = tf.reduce_mean(tf.squared_difference(Rating,loss1))
    reg=beta*(tf.norm(Q)+tf.norm(P))
    
    loss = loss+reg

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    print("STARTING THE DESCENT")

    for i in range(epochs):
        if i % 5 == 0:
            l=sess.run(loss,feed_dict={Rating:X})
            print("EPOCH " +str(i)+ ", Loss: " + str(l))
            if l<tolerance:
                break
        sess.run(train_step,feed_dict={Rating:X})
    
    q_f=sess.run(Q)
    p_f=sess.run(P)
    sess.close()
    return q_f,p_f
    
Q,P=Matrix_Factorization(R,epochs=20)