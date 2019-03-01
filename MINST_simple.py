import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load data
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)

# construct Graph
#x data
x=tf.placeholder("float", [None, 784])

#weights
W=tf.Variable(tf.zeros([784,10]))

b=tf.Variable(tf.zeros([1,10]))

#y data
y=tf.nn.softmax(tf.matmul(x,W)+b)

#predict data
y_=tf.placeholder("float", [None,10])

#cost function
cross_entropy= -tf.reduce_sum(y_*tf.log(y))

#train
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#acc_pre
correct_pre= tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
acc_pre=tf.reduce_mean(tf.cast(correct_pre,"float"))

#initial
init= tf.global_variables_initializer()

#Run Graph
sess=tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys= mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	# predict
	if i%100==0:
		print(sess.run(acc_pre, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))


print("predict acc:",sess.run(acc_pre, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
#print("weights:", sess.run(W))
print("bias", sess.run(b))
sess.close()