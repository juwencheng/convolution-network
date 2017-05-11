import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input
def neural_input(image_shape, name="x"):
    return tf.placeholder(tf.float32, [None, *image_shape], name=name)

def neural_label(n_classes, name="labels"):
    return tf.placeholder(tf.float32, [None, n_classes], name=name)

def neural_keep_prob(name="keep_prob"):
    return tf.placeholder(tf.float32, name=name)

# flatten
def flatten(input):
    shape = input.get_shape().as_list()
    width = shape[1]
    height = shape[2]
    depth = shape[3]
    flattened = tf.reshape(input, [-1, width * height * depth])
    return flattened

# conv
def conv_layer(input, conv_num_output, conv_kernel_size, conv_strides, pool_size, pool_strides):
    # input shape [None, width, height, depth]
    # so
    input_depth = input.get_shape().as_list()[3]
    w = tf.Variable(
        tf.truncated_normal([*conv_kernel_size, input_depth, conv_num_output], stddev=0.01),
        name="w")
    b = tf.Variable(tf.zeros([conv_num_output]), name="b")

    # conv
    conv = tf.nn.conv2d(input, w, strides=[1, *conv_strides, 1], padding="SAME")
    conv = tf.nn.bias_add(conv, b)

    # relu
    activation = tf.nn.relu(conv)

    # max pool
    output = tf.nn.max_pool(activation, ksize=[1, *pool_size, 1],
                                strides=[1, *pool_strides, 1], padding="SAME")

    return output

# output do not use any activation function , cause we will use softmax and cross entropy on it later
def output(input, num_outputs):
    shape = input.get_shape().as_list()
    num_inputs = shape[1]
    w = tf.Variable(
        tf.truncated_normal(shape=[num_inputs, num_outputs], stddev=0.01),
        name='w')
    b = tf.Variable(tf.zeros([num_outputs]), name='b')

    output = tf.nn.bias_add(tf.matmul(input, w), b)
    return output

# fc full-connected layer
def fc_layer(input, num_outputs):
    shape = input.get_shape().as_list()
    num_inputs = shape[1]
    w = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs], stddev=0.01), name='w')
    b = tf.Variable(tf.zeros([num_outputs]), name='b')

    fc_output = tf.nn.bias_add(tf.matmul(input, w), b)
    activation = tf.nn.relu(fc_output)
    return activation

# convolutional net
def conv_net(input, keep_prob):
    # apply conv
    conv1 = conv_layer(input, 32, (3, 3), (1, 1), (2, 2), (2, 2))

    # flatten
    flattened = flatten(conv1)

    # fully connection
    fc = fc_layer(flattened, 1024)

    # output
    logits = output(fc, 10)

    return logits

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer, feed_dict={
        x: feature_batch.reshape([-1, 28, 28, 1]),
        y: label_batch,
        keep_prob: keep_probability
    })

def print_stats(session, feature_batch, label_batch, valid_feature, valid_label, cost, accuracy):
    loss = session.run(cost, feed_dict={
        x: feature_batch.reshape([-1, 28, 28, 1]),
        y: label_batch,
        keep_prob: 1.
    })
    valid_acc = session.run(accuracy, feed_dict={
        x: valid_feature.reshape([-1, 28, 28, 1]),
        y: valid_label,
        keep_prob: 1.
    })
    print("Loss: ", loss, " Validation Accuracy: ", valid_acc)

def train(epochs=1, learning_rate=0.01, batch_size=10, keep_probability=0.9):
    # remove previous weights, bias, inputs, etc...
    # tf.reset_default_graph()

    # model
    logits = conv_net(x, keep_prob)
    logits = tf.identity(logits, name='logits')

    # loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            total_batch = mnist.train.num_examples // batch_size
            for batch_i in range(total_batch):
                feature_batch, label_batch = mnist.train.next_batch(batch_size)
                train_neural_network(sess, optimizer, keep_probability, feature_batch, label_batch)
                if batch_i % 100 == 0:
                    print_stats(sess, feature_batch, label_batch, mnist.test.images[:1000], mnist.test.labels[:1000], cost, accuracy)

        # save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./conv_model")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.329, 0.587, 0.114])

def predict():
    logits = conv_net(x, keep_prob)
    logits = tf.identity(logits, name='logits')

    # loss and optimizer
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    # optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    pred = tf.argmax(logits, 1)
    # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    # train

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./conv_model")
        img = mpimg.imread('7.png')
        # gray = rgb2gray(img)

        pred_number = sess.run(pred, feed_dict={
            x: img.reshape([-1,28, 28, 1])
        })
        print("predicted number is : ", pred_number,
            " correct number is : 1")


# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
# input
x = neural_input((28,28,1))
y = neural_label(10)
keep_prob = neural_keep_prob()

# predict()
train()
