import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)

#load data
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size=28
time_step=28
input_size_nn=28
input_size_rnn=128
units_num_nn=128
units_num_rnn=128
class_num=10

x= tf.placeholder("float32", [None, time_step, input_size_nn])

y= tf.placeholder("float32", [None, class_num])

weights= {
	'in': tf.Variable(tf.random_normal([input_size_nn, units_num_nn])),

	'out': tf.Variable(tf.random_normal([units_num_rnn, class_num]))
	}

bias= {
	'in': tf.Variable(tf.constant(0.1, shape=[units_num_nn])),

	'out': tf.Variable(tf.constant(0.1, shape=[class_num]))
	}

def LSTM(X, weights, bias):
	#layer in
	X= tf.reshape(X, [-1, input_size_nn])
	h1= tf.matmul(X, weights['in'])+ bias['in']
	h1= tf.reshape(h1, [-1, time_step, input_size_rnn])
	#for dynamic_rnn, input:= [batch, train_step, input_size]

	#LSTM cell
	lstm_cell= tf.contrib.rnn.BasicLSTMCell(units_num_rnn, forget_bias=1.0, state_is_tuple=True)#

	init_state= lstm_cell.zero_state(batch_size, dtype=tf.float32) #state:= [batch, units_num_rnn]
	outputs, final_state= tf.nn.dynamic_rnn(lstm_cell, h1, initial_state=init_state, time_major=False)
	#output= new_h:= [batch, train_step, n_unit_rnn]
	#final_state= [new_h, new_state]

	#layer out
	outputs= tf.unstack(tf.transpose(outputs, [1,0,2])) #train_step*[batch, n_unit_rnn]
	results = tf.matmul(outputs[-1], weights['out']) + bias['out'] #chose last output

	return results


pred= LSTM(x, weights, bias)
cost= tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_step= tf.train.AdamOptimizer(1e-4).minimize(cost)

correct_pre= tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc=tf.reduce_mean(tf.cast(correct_pre, "float32"))

init= tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		batch_x, batch_y= mnist.train.next_batch(batch_size)
		batch_x= batch_x.reshape([batch_size, time_step, input_size_nn])
		sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
		if i%100==0:
			batch_x_pre, batch_y_pre= mnist.test.next_batch(batch_size)
			batch_x_pre= batch_x_pre.reshape([batch_size, time_step, input_size_nn])
			print(sess.run(acc, feed_dict={x: batch_x_pre, y: batch_y_pre}))
