import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# weight
def W(shape):
	init=tf.random_normal(shape, stddev=0.1)
	return tf.Variable(init)

def b(shape):
	init=tf.zeros(shape)
	return tf.Variable(init)


#load data
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)

# feature
x= tf.placeholder("float", [None, 784])

# label
y_= tf.placeholder("float", [None, 10])

#layer 1 (100 nuerons)
W1= W([784, 100])
b1= b([100])
h1= tf.nn.relu(tf.matmul(x, W1)+ b1)

#layer 2 (10 nuerons)
W2= W([100, 10])
b2= b([10])
h2= tf.nn.softmax(tf.matmul(h1, W2)+ b2)

#cost function
cross_entropy= -tf.reduce_sum(y_*tf.log(h2))

#train step
train_step= tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# acc
acc_pre= tf.equal(tf.argmax(y_, 1), tf.argmax(h2, 1))
acc= tf.reduce_mean(tf.cast(acc_pre, "float"))

#initial
init= tf.global_variables_initializer()

# run
sess=tf.Session()
sess.run(init)

for i in range(2000):
	batch=mnist.train.next_batch(50)
	sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
	if i%100==0:
		print(sess.run(acc, feed_dict={x: batch[0], y_: batch[1]}))

print("pre", sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
sess.close()