#https://arxiv.org/pdf/1606.04080.pdf ## Matching Networks for One Shot Learning

from one_shot_learning_class import one_shot

import tensorflow as tf #version 1.4
import numpy as np
import get_S_B_MNIST as sb
import os


def train(model, data):
	batch_size = 32#128#len(data)##32
	loss = 0
	
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		#print(i+1,',', int(np.ceil(len(data)/batch_size)))
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		train_loss, _ = sess.run([model.cost, model.minimize], {model.X:input_, model.Y:target_, model.is_train:True})
		loss += train_loss
	
	return loss / len(data)


def validation(model, data):
	batch_size = 512#32
	loss = 0
	
	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]
	
		vali_loss = sess.run(model.cost, {model.X:input_, model.Y:target_, model.is_train:False})
		loss += vali_loss
	
	return loss / len(data)


def test(model, data):
	batch_size = 512#32
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784]
		target_ = batch[:, 784:]

		check = sess.run(model.correct, {model.X:input_, model.Y:target_, model.is_train:False})
		correct += check

	return correct / len(data)


def run(model, train_set, vali_set, test_set, restore=0):
	#restore인지 체크.
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
	
	for epoch in range(1, 4001):
		train_loss = train(model, train_set)
		vali_loss = validation(model, vali_set)
		accuracy_t = test(model, train_set)
		accuracy = test(model, test_set)
		
		print("epoch : ", epoch, "\t train_loss : ", train_loss, "\t vali_loss : ", vali_loss, "\t train accuracy : ", accuracy_t, "\t test accuracy : ", accuracy)
		
		


sess = tf.Session()

#S: support set(메모리 역할), B: batch set(학습에 사용)

shot = 5
batch_set_num = 5 # #data per class
vali_set_num = 10

S, B, vali_set = sb.get_data(shot=shot, batch_set_num=batch_set_num, vali_set_num=vali_set_num) #class가 10개니깐 각각 10배씩 뽑힘.
test_set = sb.get_test_data()

model = one_shot(sess, S, K=8, train_rate=0.00008)
run(model, B, vali_set, test_set)
