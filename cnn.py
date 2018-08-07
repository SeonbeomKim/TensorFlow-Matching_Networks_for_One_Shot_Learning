import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import get_S_B_MNIST as sb
import math

train_rate = 0.001
height = 28
width = 28
channel = 1 #mnist is 1 color

def train(data):
	batch_size = 32#128
	loss = 0
	np.random.shuffle(data)

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		#print(i+1, '/', int(math.ceil(len(data)/batch_size)))
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		train_loss, _ = sess.run([cross_entropy, minimize], {X:input_, Y:target_, is_train:True})
		loss += train_loss
	
	return loss


def validation(data):
	batch_size = 512#512
	loss = 0
	
	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		vali_loss = sess.run(cross_entropy, {X:input_, Y:target_, is_train:False})
		loss += vali_loss
	
	return loss


def test(data):
	batch_size = 512#512
	correct = 0

	for i in range( int(math.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]

		check = sess.run(correct_check, {X:input_, Y:target_, is_train:False})
		correct += check

	return correct / len(data)


def run(train_set, vali_set, test_set):
	for epoch in range(1, 301):
		train_loss = train(train_set)
		vali_loss = validation(vali_set)
		accuracy = test(test_set)

		print("epoch : ", epoch, " train_loss : ", train_loss, " vali_loss : ", vali_loss, " accuracy : ", accuracy)

		

with tf.device('/gpu:0'):
	X = tf.placeholder(tf.float32, [None, 784]) #batch
	Y = tf.placeholder(tf.float32, [None, 10]) #batch
	X_reshape = tf.reshape(X, (-1, height, width, channel))
	
	is_train = tf.placeholder(tf.bool)
	activation = tf.nn.relu

	layer1 = tf.layers.conv2d(X_reshape, filters=256, kernel_size = [3,3], strides=[1, 1], padding='SAME') 
	bn_layer1 = tf.layers.batch_normalization(layer1, training=is_train)
	layer1 = activation(bn_layer1)	
	pool_layer1 = tf.layers.max_pooling2d(layer1, pool_size = [2, 2], strides=[2, 2], padding='SAME') 

	layer2 = tf.layers.conv2d(pool_layer1, filters=512, kernel_size = [3,3], strides=[1, 1], padding='SAME') 
	bn_layer2 = tf.layers.batch_normalization(layer2, training=is_train)
	layer2 = activation(bn_layer2)
	pool_layer2 = tf.layers.max_pooling2d(layer2, pool_size = [2, 2], strides=[2, 2], padding='SAME')
	
	layer3 = tf.layers.conv2d(pool_layer2, filters=1024, kernel_size = [3,3], strides=[1, 1], padding='SAME') 
	bn_layer3 = tf.layers.batch_normalization(layer3, training=is_train)
	layer3 = activation(bn_layer3)
	pool_layer3 = tf.layers.max_pooling2d(layer3, pool_size = [2, 2], strides=[2, 2], padding='SAME') 

	flat = tf.layers.flatten(pool_layer3) # batch, 4*4*256

	FC = tf.layers.dense(flat, units = 2048)#, activation=None) # batch, self.cell_num
	bn_FC = tf.layers.batch_normalization(FC, training=is_train)
	FC = activation(bn_FC)
	
	output = tf.layers.dense(FC, units = 10)#, activation=None) # batch, self.cell_num
	bn_output = tf.layers.batch_normalization(output, training=is_train)
	output = bn_output

	cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = output) )	

	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		minimize = tf.train.AdamOptimizer(train_rate).minimize(cross_entropy)

	correct_check = tf.reduce_sum(tf.cast( tf.equal( tf.argmax(output, 1), tf.argmax(Y, 1) ), tf.int32 ))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

shot = 5 + 5
batch_set_num = 0 #each class
vali_set_num = 10

train_set, _, vali_set = sb.get_data(shot=shot, batch_set_num=batch_set_num, vali_set_num=vali_set_num) #class가 10개니깐 각각 10배씩 뽑힘.
test_set = sb.get_test_data()
#print(train_set.shape, vali_set.shape, test_set.shape)


run(train_set, vali_set, test_set)
