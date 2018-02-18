
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tempfile
import tensorflow as tf
import sys
import numpy
import math
# Number of classes is 2 (squares and triangles)
from tensorflow.python.framework.errors_impl import InvalidArgumentError
FLAGS = None
nClass=6

# Simple model (set to True) or convolutional neural network (set to False)
simpleModel=False
fullFNN=False
# Dimensions of image (pixels)
height=640
width=480
NUM_CLASSES=6
# The MNIST images are always 28x28 pixels.
IMAGE_PIXELS = height * width
hidden1_units=256
hidden2_units=64

# Function to tell TensorFlow how to read a single image from input file
def getImage(filename):
	# convert filenames to a queue for an input pipeline.
	filenameQ = tf.train.string_input_producer([filename],num_epochs=None)

	# object to read records
	recordReader = tf.TFRecordReader()

	# read the full set of features for a single example 
	key, fullExample = recordReader.read(filenameQ)

	# parse the full example into its' component features.
	try:
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
	except(InvalidArgumentError):
	    raise ValueError


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

def main():
	new_saver = tf.train.import_meta_graph('my-model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	all_vars = tf.get_collection('vars')
	for v in all_vars:
	    v_ = sess.run(v)
	    print(v_)
    noValue,image=getImage(FLAGS.picture_file)
    feed_dict = {x: [image]}
	classification = tf.run(v, feed_dict)
	print( classification)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--picture_file', type=str,
	                  default='pcdFileToUse.JPEG ',
	                  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)