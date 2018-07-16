import tensorflow as tf
import sys,subprocess
import numpy,os
import math,pickle


# Number of classes is 2 (squares and triangles)
from tensorflow.python.framework.errors_impl import InvalidArgumentError
nClass=5

# Simple model (set to True) or convolutional neural network (set to False)
simpleModel=False
fullFNN=False
# Dimensions of image (pixels)
height=40
width=40
NUM_CLASSES=5
IMAGE_PIXELS = height * width
hidden1_units=64
hidden2_units=32



def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
# Function to tell TensorFlow how to read a single image from input file
# Function to tell TensorFlow how to read a single image from input file
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

    image=tf.reshape(tf.image.rgb_to_grayscale(image),[height*width])

    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, nClass))

    return label, image



#saver = tf.train.Saver()



def makeModel(keep_prob,keep_prob2,x):
  # run convolutional neural network model given in "Expert MNIST" TensorFlow tutorial
  def conv_layer(nFeatures,ker_size,input_layer,inputSize):

    W_conv1 = weight_variable([ker_size, ker_size, inputSize, nFeatures])
    b_conv1 = bias_variable([nFeatures])
    return tf.nn.relu(conv2d(input_layer, W_conv1) + b_conv1)
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
  nFeatures1=16
  nFeatures2=32
  nFeatures3=64
  nNeuronsfc=120
  nNeuronsfc2=60

  # use functions to init weights and biases
  # nFeatures1 features for each patch of size 5x5
  # SAME weights used for all patches
  # 1 input channel
  
  # reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth 
  # means 1 colour channel per pixel
  # x_image = tf.reshape(x, [-1,28,28,1])
  x_image = tf.reshape(x, [-1,width,height,1])
  
  
  # hidden layer 1 
  # pool(convolution(Wx)+b)
  # pool reduces each dim by factor of 2.
  h_conv1 = conv_layer(nFeatures1,4,x_image,1)
  h_conv1mid = conv_layer(nFeatures1,3,h_conv1,nFeatures1)
  h_pool1 = max_pool_2x2(h_conv1mid)
  
  # similarly for second layer, with nFeatures2 features per 5x5 patch
  # input is nFeatures1 (number of features output from previous layer)
  h_conv2 = conv_layer(nFeatures2,4,h_pool1,nFeatures1)
  h_conv2mid = conv_layer(nFeatures2,3,h_conv2,nFeatures2)
  h_pool2 = max_pool_2x2(h_conv2mid)
  
  h_conv3 = conv_layer(nFeatures3,4,h_pool2,nFeatures2)
  h_conv3mid = conv_layer(nFeatures3,2,h_conv3,nFeatures3)
  h_pool3 = max_pool_2x2(h_conv3mid)
  
  # denseley connected layer. Similar to above, but operating
  # on entire image (rather than patch) which has been reduced by a factor of 4 
  # in each dimension
  # so use large number of neurons 

  W_fc1 = weight_variable([int(math.floor(width/8) * math.floor(height/8)) * nFeatures3, nNeuronsfc])
  b_fc1 = bias_variable([nNeuronsfc])
  
  # flatten output from previous layer
  h_pool2_flat = tf.reshape(h_pool3, [-1, int(math.floor(width/8) * math.floor(height/8)) * nFeatures3])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  # reduce overfitting by applying dropout
  # each neuron is kept with probability keep_prob
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


  W_fc1mid = weight_variable([nNeuronsfc, nNeuronsfc2])
  b_fc1mid = bias_variable([nNeuronsfc2])

  h_fc1mid = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc1mid) + b_fc1mid)


  h_fc1_drop_mid = tf.nn.dropout(h_fc1mid, keep_prob2)

  # create readout layer which outputs to nClass categories
  W_fc2 = weight_variable([nNeuronsfc2, nClass])
  b_fc2 = bias_variable([nClass])
  
  # define output calc (for each class) y = softmax(Wx+b)
  # softmax gives probability distribution across all classes
  # this is not run until later
  y = tf.nn.softmax(tf.matmul(h_fc1_drop_mid, W_fc2) + b_fc2,name='this_is_very_special')
  return y





# start training
def runTrain(nSteps,chenSteps):
  # run the session

# initialize the variables
# associate the "label" and "image" objects with the corresponding features read from 
# a single example in the training data file
  label, image = getImage("/home/jase/Downloads/train-00000-of-00001")

# and similarly for the validation data
  vlabel, vimage = getImage("/home/jase/Downloads/validation-00000-of-00002")


  
# associate the "label_batch" and "image_batch" objects with a randomly selected batch---
# of labels and images respectively
  imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label], batch_size=10,
    capacity=569646,
    min_after_dequeue=7500)#515133

# and similarly for the validation data 
  vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], batch_size=100,
    capacity=int(165000/2),
    min_after_dequeue=1000)#90360


  clabel, cimage = getImage("/home/jase/UploadFolder/train-00000-of-00001")

# and similarly for the validation data
  cvlabel, cvimage = getImage("/home/jase/UploadFolder/validation-00000-of-00002")

  ctlabel, ctimage = getImage("/home/jase/UploadFolder/validation-00001-of-00002")

  
# associate the "label_batch" and "image_batch" objects with a randomly selected batch---
# of labels and images respectively
  cimageBatch, clabelBatch = tf.train.shuffle_batch(
    [cimage, clabel], batch_size=10,
    capacity=165000,
    min_after_dequeue=7500)#515133

# and similarly for the validation data 
  cvimageBatch, cvlabelBatch = tf.train.shuffle_batch(
    [cvimage, cvlabel], batch_size=100,
    capacity=int(165000/20),
    min_after_dequeue=1000)#90360

  ctimageBatch, ctlabelBatch = tf.train.shuffle_batch(
    [ctimage, ctlabel], batch_size=5000,
    capacity=int(165000/20),
    min_after_dequeue=1000)# interactive session allows inteleaving of building and running steps
  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, [None, width*height])
# similarly, we have a placeholder for true outputs (obtained from labels)
  y_ = tf.placeholder(tf.float32, [None, nClass])

  keep_prob = tf.placeholder(tf.float32)
  keep_prob2 = tf.placeholder(tf.float32)

  y=makeModel(keep_prob,keep_prob2,x)

# measure of error of our model
# this needs to be minimised by adjusting W and b
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step which minimises cross entropy
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
  correct_prediction = tf.equal(tf.argmax(y,1,name='weeee'), tf.argmax(y_,1))

# get mean of all entries in correct prediction, the higher the better
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# start the threads used for reading files
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  sess.run(tf.global_variables_initializer())

  for i in range(nSteps):
  
      batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
  
      # run the training step with feed of images
      if simpleModel:
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
      else:
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8, keep_prob2: 0.5})
  
  
      if (i+1)%100 == 0: # then perform validation
        totalAcc=0.0
        for j in range(1):
        # get a validation batch
          vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
          if simpleModel:
            train_accuracy = accuracy.eval(feed_dict={
              x:vbatch_xs, y_: vbatch_ys})
          else:
            train_accuracy = accuracy.eval(feed_dict={
              x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, keep_prob2: 1.0})
          print("step %d, simu training accuracy %g"%(i+1, train_accuracy))
  

  for i in range(chenSteps):
  
      batch_xs, batch_ys = sess.run([cimageBatch, clabelBatch])
  
      # run the training step with feed of images
      if simpleModel:
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
      else:
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8, keep_prob2: 0.5})
  
  
      if (i+1)%100 == 0: # then perform validation
        totalAcc=0.0
        for j in range(1):
        # get a validation batch
          vbatch_xs, vbatch_ys = sess.run([cvimageBatch, cvlabelBatch])
          if simpleModel:
            train_accuracy = accuracy.eval(feed_dict={
              x:vbatch_xs, y_: vbatch_ys})
          else:
            train_accuracy = accuracy.eval(feed_dict={
              x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0, keep_prob2: 1.0})
          print("step %d, chen training accuracy %g"%(i+1, train_accuracy))


  tbatch_xs, tbatch_ys = sess.run([ctimageBatch, ctlabelBatch])
  if simpleModel:
    train_accuracy = accuracy.eval(feed_dict={
      x:tbatch_xs, y_: tbatch_ys})
  else:
    train_accuracy = accuracy.eval(feed_dict={
      x:tbatch_xs, y_: tbatch_ys, keep_prob: 1.0 , keep_prob2: 1.0})
  print("step %d, chen test accuracy %g"%(0, train_accuracy))
  coord.request_stop()
  coord.join(threads)
  sess.close()
  return 1*train_accuracy

'''export_path_base = '/home/jasebroderick/Documents/MSThesis/'
export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes('CNNModelSavedToUse6'))
print ('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess,[tf.saved_model.tag_constants.SERVING])
builder.save()'''
for k in range(10):
  classStepsTOAcc={}
  for i in range(20):
    for j in range(20):
      classStepsTOAcc[i,j]=runTrain(i*500,j*500)
  save_object(classStepsTOAcc,'tim_donecun'+str(k)+'.pkl')
