
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import pandas as pd
#from sklearn.decomposition import PCA
#from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
#import math as math

f = open('results.txt','a')
f1 = open('step.txt','a')
f2 = open('loss.txt','a')
f3 = open('train.txt','a')
f4 = open('validation.txt','a')
f.write('\n genes + mutation - chromosome - pca')
f1.write('\n [')
f2.write('\n [')
f3.write('\n [')
f4.write('\n [')
def split_traintest():
    haward = pd.read_csv('~/Documents/storage/aichallenge/GTD_0718dist/GTD.csv', sep=',',index_col=0)
    #haward=haward.drop(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'], axis=1)
    colsize = len(list(haward))-1
    #print(len(haward['primary'].tolist()))
    PRIMARY = haward.gname.unique().tolist()
    f.write("The shape of the data is: "+str(haward.shape)) #print(PRIMARY)
    #haward = haward.drop("tumour", axis=1)
    haward.gname = pd.Categorical(haward.gname).codes
    haward = haward.astype(np.float32)
    #print(haward.gname)
    #divide train and test
    #haward.head(3)
    msk1 = np.random.rand(len(haward)) < 0.6
    train = haward[msk1]
    train_dataset = train.loc[:, train.columns != 'gname']
    train_labels = train.loc[:, train.columns == 'gname']
    testandvalid = haward[~msk1]
    msk2 = np.random.rand(len(testandvalid)) < 0.5
    valid = testandvalid[msk2]
    valid_dataset = valid.loc[:, valid.columns != 'gname']	
    valid_labels = valid.loc[:, valid.columns == 'gname']
    test = testandvalid[~msk2]
    test_dataset = test.loc[:, test.columns != 'gname']
    test_labels = test.loc[:, test.columns == 'gname']
    del haward # delete to free space
    del train
    del testandvalid
    del valid
    del test
    return train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels, colsize
train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels,colsize = split_traintest()
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



'''print(train_labels)''' 
####reformat data

colsize = colsize
num_labels = 5

def reformat(dataset,labels):
  dataset=dataset.values
  labels = pd.get_dummies(labels)
  lst=labels['gname'].tolist()
  k=pd.Series(lst)
  labels = pd.get_dummies(k).values
  return dataset,labels

train_dataset, train_labels = reformat(train_dataset,train_labels)
valid_dataset, valid_labels = reformat(valid_dataset,valid_labels)
test_dataset, test_labels = reformat(test_dataset,test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# This is to expedite the process 
train_subset = 10000
# This is a good beta value to start with
beta = 0.01

graph = tf.Graph()
with graph.as_default():

    # Input data.
    # They're all constants.
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
  
    # Variables    
    # They are variables we want to update and optimize.
    weights = tf.Variable(tf.truncated_normal([colsize, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
  
    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases 
    # Original loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels) )
    # Loss function using L2 Regularization
    regularizer = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss + beta * regularizer)
    
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax( tf.matmul(tf_valid_dataset, weights) + biases )
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    


    #Run Computation & Iterate

num_steps = 10001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
def confusion(predictions, labels):
    from sklearn import metrics
    #print(metrics.accuracy_score(labels, predictions)) 
    #return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)))
    return(metrics.confusion_matrix(np.argmax(labels, 1), np.argmax(predictions, 1)))

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases. 
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step {}: {}'.format(step, l))
            print('Training accuracy: {:.1f}'.format(accuracy(predictions, 
                                                         train_labels[:train_subset, :])))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            
            # You don't have to do .eval above because we already ran the session for the
            # train_prediction
            print('Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(),valid_labels)))
            f1.write('{},'.format(step))
            f2.write('{},'.format(l))
            f3.write('{:.1f},'.format(accuracy(predictions, train_labels[:train_subset, :])))
            f4.write('{:.1f},'.format(accuracy(valid_prediction.eval(), valid_labels)))


    #here we make the confusion Matrix.
    print(np.argmax(test_prediction.eval(),1))
    print(np.argmax(test_labels,1))
    print('Test accuracy: {:.1f}'.format(accuracy(test_prediction.eval(), test_labels))) 
    print('confusion:', confusion(test_prediction.eval(), test_labels))
    f.write('\n Training accuracy: {:.1f}'.format(accuracy(predictions, train_labels[:train_subset, :])))
    f.write('\n Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(),valid_labels)))
    f.write('\n Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(),valid_labels)))
    f.write('\n The confusion matrix is  confusion:\n')
    #cfn = confusion(test_prediction.eval(), test_labels)
    #np.savetxt('confusion.txt', cfn, delimiter=" ", fmt="%s")#f.write(confusion(test_prediction.eval(), test_labels))

f1.write(']')
f2.write(']')
f3.write(']')
f4.write(']')

f1.close()
f2.close()
f3.close()
f4.close()
f.close()
###########################################################################################
'''THE END'''
###########################################################################################

        


