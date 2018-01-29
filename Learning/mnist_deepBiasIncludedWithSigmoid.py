# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import math
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
batch_size = 20            # Batch size for stochastic gradient descent
test_size = batch_size      # Temporary heuristic. In future we'd like to decouple testing from batching
max_iterations = 200000       # Max number of iterations
learning_rate = 1e-4        # Learning rate
num_classes = 10            # Number of target classes, 10 for MNIST
centers = [784,350]             # Number of "hidden neurons" 1 that is number of centroids [784,500,400,300,200,100,50]     
var_rbf = 225 # What variance do you expect workable for the RBF?


def rbf(x):
    prevLayer=x
    for j in range(1,len(centers)):
    	prevLayerNum=centers[j-1]
    	num_center = centers[j]
    	with tf.name_scope("Hidden_layer" + str(j)) as scope:
            w = tf.Variable(tf.truncated_normal([centers[j-1], num_center], stddev=1.0/math.sqrt(float(centers[j-1])), dtype=tf.float32),name='weight' + str(j))
            bias = tf.Variable(tf.constant(1.1, shape=[num_center]), name='bias' + str(j))
            h=tf.matmul(prevLayer, w) + bias
            prevLayerTemp=rbf_activation(num_center,prevLayer,prevLayerNum)

            w2 = tf.Variable(tf.truncated_normal([num_center, num_center], stddev=1.0/math.sqrt(float(centers[j-1])), dtype=tf.float32),name='weight2' + str(j))
            bias2 = tf.Variable(tf.constant(1.1, shape=[num_center]), name='bias2' + str(j))
            h2=tf.matmul(prevLayerTemp,w2)+bias2
            prevLayer2=tf.nn.relu(h2)

            w3 = tf.Variable(tf.truncated_normal([num_center, num_center], stddev=1.0/math.sqrt(float(centers[j-1])), dtype=tf.float32),name='weight3' + str(j))
            bias3 = tf.Variable(tf.constant(1.1, shape=[num_center]), name='bias3' + str(j))
            h3=tf.matmul(prevLayer2,w3)+bias3
            prevLayer=tf.nn.relu(h3)
    with tf.name_scope('softmax_linear'):
	    weights = tf.Variable(
	        tf.truncated_normal([centers[-1], num_classes],
	                            stddev=1.0 / math.sqrt(float(centers[-1]))),
	        name='weights')
	    biases = tf.Variable(tf.zeros([num_classes]),
	                         name='biases')
	    logits = tf.matmul(prevLayer, weights) + biases
	    return logits
def rbf_activation(num_center,prevLayer,prevLayerNum):
    #RBF Quadractic Activation Function 
    centroids = tf.Variable(tf.random_uniform([num_center,prevLayerNum],dtype=tf.float32),name='centroids')
    var = tf.Variable(tf.truncated_normal([num_center],mean=var_rbf,stddev=5,dtype=tf.float32),name='RBF_variance')#For now, we collect the distanc
    exp_list = []
    for i in range(num_center):
    	exp_list.append((-1*tf.reduce_sum(tf.square(tf.subtract(prevLayer,centroids[i,:])),1)/(2*var[i])))
    activatedPeviousLayer = tf.transpose(tf.stack(exp_list))
    return activatedPeviousLayer



def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  logits = rbf(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=logits)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_iterations):
      batch = mnist.train.next_batch(batch_size)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
