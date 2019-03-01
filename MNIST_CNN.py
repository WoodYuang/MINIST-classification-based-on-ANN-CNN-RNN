import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def w_variable(shape):
	init=tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init)

def b_variable(shape):
	init=tf.constant(0.1, shape=shape)
	return tf.Variable(init)

def conv2d(x, W): #x input, W filter
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') #size W=[length, high, in_channel, out_channel]

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



#load data
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)

x= tf.placeholder("float", shape=[None, 784])

y_= tf.placeholder("float", shape=[None, 10])

x_image= tf.reshape(x, [-1,28,28,1]) #size [num, length, high, channel]

# 1th layer
W_covn1= w_variable([5,5,1,32])  #the covn core W
b_covn1= b_variable([32])


h_conv1= tf.nn.relu(conv2d(x_image, W_covn1)+ b_covn1)
h_pool1= max_pool_2x2(h_conv1)

# 2th layer
W_covn2= w_variable([5,5,32,64])
b_covn2= b_variable([64])

h_conv2= tf.nn.relu(conv2d(h_pool1, W_covn2)+ b_covn2)
h_pool2= max_pool_2x2(h_conv2)

# full connect layer
W_fc1= w_variable([7*7*64, 1024])
b_fc1= b_variable([1024])

h_pool2_flat= tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1= tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+ b_fc1)

# drop out
keep_prob=tf.placeholder("float")
h_pool2_flat= tf.nn.dropout(h_fc1, keep_prob)

#out layer
W_fc2= w_variable([1024, 10])
b_fc2= b_variable([10])

h_fc2= tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc2)+ b_fc2)

# loss fuction
cross_entropy= -tf.reduce_sum(y_*tf.log(h_fc2))

# train
train_step= tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#pre acc
correct_prediction= tf.equal(tf.argmax(y_, 1), tf.argmax(h_fc2, 1))
acc= tf.reduce_mean(tf.cast(correct_prediction, "float"))

# init
init=tf.global_variables_initializer()

#run
sess=tf.Session()
sess.run(init)

for i in range(20000):
	batch= mnist.train.next_batch(50)
	#sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	if i%100 == 0:
		acc_train= sess.run(acc, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
		print("step:", i)
		print("acc_train:", acc_train)


print("acc_pre",sess.run(acc, feed_dict={y_: mnist.test.images, h_fc2: mnist.test.labels, keep_prob: 1.0}))
sess.close()