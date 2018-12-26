import pandas as pd
import tensorflow as tf
import numpy as np
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler

##Write the data.
#finalDf.to_csv("~/haward/ket/newdata/clean20kfinalDF.csv", sep='\t', encoding='utf-8')

#read your data
#haward = pd.read_csv("~/Desktop/workingarea/data/newdata/k15.csv",sep="\t",index_col=0)
#print(list(haward))
def split_traintest(p):
    haward = pd.read_csv('~/Documents/storage/aichallenge/GTD_0718dist/GTD.csv', sep=',',index_col=0)
    haward = haward.drop(["attacktype1_txt","weaptype1_txt"], axis=1)
    print(haward.shape)
    print(len(haward['gname'].tolist()))
    PRIMARY = haward.gname.unique().tolist()
    #haward = haward.drop("tumour", axis=1)
    haward.gname = pd.Categorical(haward.gname).codes
    haward = haward.astype(np.int64)
    #divide train and test
    msk = np.random.rand(len(haward)) < 0.6
    train = haward[msk]
    test = haward[~msk]
    return train,test

#CSV_COLUMN_NAMES = list(maybe_dopca(p))#['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'primary']
#PRIMARY = ['Setosa', 'Versicolor', 'Virginica']
#PRIMARY = df.primary.unique().tolist()
'''def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path'''
def load_data(p,y_name='gname'):
    """Returns the cancer dataset as (train_x, train_y), (test_x, test_y)."""
    #train_path, test_path = maybe_download()
    train,test = split_traintest(p)
    #train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES,sep="\t", header=0)
    #print (list(train))
    train_x, train_y = train, train.pop(y_name)

    #test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES,sep="\t", header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)
'''def load_data(y_name='primary'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)'''


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('gname')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
