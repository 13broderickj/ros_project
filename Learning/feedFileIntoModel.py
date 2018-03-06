
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tempfile
import tensorflow as tf
import sys,os
import numpy
import math
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file

# Number of classes is 2 (squares and triangles)
from tensorflow.python.framework.errors_impl import InvalidArgumentError
FLAGS = None
nClass=6

# Dimensions of image (pixels)
height=36
width=60
parser = argparse.ArgumentParser()
parser.add_argument('--picture_file', type=str,
	                  default='pcdFileToUse.JPEG ',
	                  help='Directory for storing input data')

# this creates a placeholder for x, to be populated later

def getImage(filename):
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename],num_epochs=None)
 
    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example 
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/channels':  tf.FixedLenFeature([], tf.int64),            
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })


    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg',[image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)
    
        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel. 
    # the "1-.." part inverts the image, so that the background is black.

    image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[height*width])

    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, nClass))

    return label, image
tlabel, timage = getImage("data/validation-00001-of-00002")
timageBatch, tlabelBatch = tf.train.shuffle_batch(
        [timage, tlabel], batch_size=1700,
        capacity=1700,
        min_after_dequeue=1000)
# x is the input array, which will contain the data from an image 
# this creates a placeholder for x, to be populated later
sess=tf.Session()

'''
if True:
  # run convolutional neural network model given in "Expert MNIST" TensorFlow tutorial

  # functions to init small positive weights and biases
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  # set up "vanilla" versions of convolution and pooling
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  print ("Running Convolutional Neural Network Model")
  nFeatures1=32
  nFeatures2=64
  nNeuronsfc=1024

  # use functions to init weights and biases
  # nFeatures1 features for each patch of size 5x5
  # SAME weights used for all patches
  # 1 input channel
  W_conv1 = weight_variable([5, 5, 1, nFeatures1])
  b_conv1 = bias_variable([nFeatures1])
  
  # reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth 
  # means 1 colour channel per pixel
  # x_image = tf.reshape(x, [-1,28,28,1])
  x_image = tf.reshape(x, [-1,width,height,1])
  
  
  # hidden layer 1 
  # pool(convolution(Wx)+b)
  # pool reduces each dim by factor of 2.
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  
  # similarly for second layer, with nFeatures2 features per 5x5 patch
  # input is nFeatures1 (number of features output from previous layer)
  W_conv2 = weight_variable([5, 5, nFeatures1, nFeatures2])
  b_conv2 = bias_variable([nFeatures2])
  

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  
  
  # denseley connected layer. Similar to above, but operating
  # on entire image (rather than patch) which has been reduced by a factor of 4 
  # in each dimension
  # so use large number of neurons 


  W_fc1 = weight_variable([math.floor(width/4) * math.floor(height/4) * nFeatures2, nNeuronsfc])
  b_fc1 = bias_variable([nNeuronsfc])
  
  # flatten output from previous layer
  h_pool2_flat = tf.reshape(h_pool2, [-1, math.floor(width/4) * math.floor(height/4) * nFeatures2])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  # reduce overfitting by applying dropout
  # each neuron is kept with probability keep_prob
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  # create readout layer which outputs to nClass categories
  W_fc2 = weight_variable([nNeuronsfc, nClass])
  b_fc2 = bias_variable([nClass])
  
  # define output calc (for each class) y = softmax(Wx+b)
  # softmax gives probability distribution across all classes
  # this is not run until later
  y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)'''
# x is the input array, which will contain the data from an image 
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
# measure of error of our model




def _process_image(filename):
	"""Process a single image file.

	Args:
	filename: string, path to an image file e.g., '/path/to/example.JPG'.
	coder: instance of ImageCoder to provide TensorFlow image coding utils.
	Returns:
	image_buffer: string, JPEG encoding of RGB image.
	height: integer, image height in pixels.
	width: integer, image width in pixels.
	"""
	  # Create a generic TensorFlow-based utility for converting all image codings.
	coder = ImageCoder()
	# Read the image file.
	with tf.gfile.FastGFile(filename, 'rb') as f:
		image_data = f.read()

	image_data = coder.png_to_jpeg(image_data)

	# Decode the RGB JPEG.
	image = coder.decode_jpeg(image_data)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[height*width])
	return image
result=None
def main():
    pathToCheck='/tmp/model2.ckpt'

    # similarly, we have a placeholder for true outputs (obtained from labels)
    sess.as_default()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(pathToCheck+'.meta')
    saver.restore(sess, pathToCheck)
    graph = tf.get_default_graph()
    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)
    chkp.print_tensors_in_checkpoint_file(pathToCheck,tensor_name='', all_tensors=True)
    image=_process_image(FLAGS.picture_file)
    print("Alll the variables printed above")
    y = graph.get_tensor_by_name("this_is_very_special:0")
    	# initialize the variables
    x=graph.get_tensor_by_name('Placeholder:0')
    y_=graph.get_tensor_by_name('Placeholder_1:0')
    keep_prob=graph.get_tensor_by_name('Placeholder_2:0')
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    # get mean of all entries in correct prediction, the higher the better
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# run the session

	# initialize the variables


	# start the threads used for reading files
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    tbatch_xs, tbatch_ys = sess.run([timageBatch, tlabelBatch])
    xToEval = sess.run([image])
    print(y.eval(feed_dict={x : tbatch_xs, keep_prob: 1.0},session=sess))
    print("Classifier says : ")
    print(y.eval(feed_dict={x : xToEval, keep_prob: 1.0},session=sess))
    #y.eval(timageBatch,session=sess)
    train_accuracy = accuracy.eval(feed_dict={
           x : tbatch_xs , y_ : tbatch_ys, keep_prob: 1.0 }, session=sess )
    print("Test Accuracy :  ",train_accuracy)
    print("All done")
    print(FLAGS.picture_file)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--picture_file', type=str,
	                  default='x_Coffee_Mug2.JPEG ',
	                  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	main()