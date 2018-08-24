import numpy as np
import pandas as pd
import os
import tensorflow as tf

def read_data(data_path, split="train"):

    num_steps = 128

    path_ = os.path.join(data_path, split)
    path_signals = os.path.join(path_, "Inertial_Signals")

    label_path = os.path.join(path_, "y_" + split + ".txt")
    labels = pd.read_csv(label_path, header=None)

    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = len(channel_files)
    posix = len(split) + 5

    list_of_channels = []
    X = np.zeros((len(labels), num_steps, n_channels))
    i_ch = 0
    for fil_ch in channel_files:
        channel_name = fil_ch[:-posix]
        dat_ = pd.read_csv(os.path.join(path_signals, fil_ch), delim_whitespace=True, header=None)
        X[:, :, i_ch] = dat_.as_matrix()

        list_of_channels.append(channel_name)

        i_ch += 1

    return X, labels[0].values

def one_hot(labels, n_class=2):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y

def cConvLay(input_, filter_):

    layer = tf.layers.conv1d(inputs=input_,
                             filters=filter_,
                             kernel_size=2,
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)
    layer = tf.layers.max_pooling1d(inputs=layer,
                                    pool_size=2,
                                    strides=2,
                                    padding='same')

    layer = tf.nn.relu(layer)

    return layer

def cFlayLay(layer):

    layer = tf.reshape(layer, (-1, 8 * 160))

    return layer

def batches(X, y, batch_size=100):
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]

def main():

    batch_size = 600
    num_channels = 9
    seq_len = 128
    learning_rate = 0.0001
    epochs = 100

    classes = ['WALKING','JOGGING']
    num_classes = len(classes)

    X_train, labels_train = read_data(data_path="./data/", split="train")

    X_train = (X_train - np.mean(X_train, axis=0)[None, :, :]) / np.std(X_train, axis=0)[None, :, :]

    y_tr = one_hot(labels_train)

    session = tf.Session()
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, num_channels], name='inputs')

    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

    convLayer1 = cConvLay(inputs_, 20)
    convLayer2 = cConvLay(convLayer1,40)
    convLayer3= cConvLay(convLayer2,80)
    convLayer4= cConvLay(convLayer3,160)

    flatLayer = cFlayLay(convLayer4)

    logits = tf.layers.dense(flatLayer, num_classes)

    y_pred = tf.nn.softmax(logits,name='y_pred')
    session.run(tf.global_variables_initializer())

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true, name='y_pred'))

    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

    session.run(tf.global_variables_initializer())

    trainA = []
    trainL = []

    saver = tf.train.Saver()

    session.run(tf.global_variables_initializer())

    for e in range(epochs):
        print("Epoch: {}/{}".format(e,epochs))
        for x, y in batches(X_train, y_tr, batch_size):

            feed = {inputs_ : x,
                    y_true : y,
                    learning_rate_ : learning_rate}

            loss, _ , acc = session.run([cost, optimizer, accuracy], feed_dict=feed)
            trainA.append(acc)
            trainL.append(loss)

    saver.save(session, "checkpoints-cnn/har")
    tf.train.write_graph(session.graph_def, '.', 'checkpoints-cnn/har.pbtxt', as_text=False)

    X_test, labels_test = read_data(data_path="./data/", split="test")

    X_test = (X_test - np.mean(X_test, axis=0)[None, :, :]) / np.std(X_test, axis=0)[None, :, :]
    y_test = one_hot(labels_test)

    test_acc = []

    saver.restore(session, tf.train.latest_checkpoint('checkpoints-cnn'))

    for x_t, y_t in batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                y_true: y_t}

        batch_acc = session.run(accuracy, feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

if __name__ == '__main__':
    main()
