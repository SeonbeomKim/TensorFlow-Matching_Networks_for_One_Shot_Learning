import tensorflow as tf #version 1.4
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def get_data(shot=1, batch_set_num=10, vali_set_num=10):
	data = np.hstack((mnist.train.images, mnist.train.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.

	check = np.zeros(10)
	dic = {}
	
	index = 0
	while np.sum(check) != 10 * (shot + batch_set_num + vali_set_num):  #라벨당 batch_set_num + shot개씩 
		label = np.argmax(data[index][784:])
		if label not in dic:
			dic[label] = [data[index]]
			check[label] += 1
		elif len(dic[label]) < shot + batch_set_num + vali_set_num:
			dic[label] = np.concatenate((dic[label], [data[index]]))
			check[label] += 1
		index += 1

	S = np.concatenate( [ dic[i][0:shot] for i in range(10) ] ) 
	B = np.concatenate( [ dic[i][shot:shot+batch_set_num] for i in range(10) ] )
	V = np.concatenate( [ dic[i][shot+batch_set_num:] for i in range(10) ] )

	return S, B, V

def get_test_data():
	test = np.hstack((mnist.test.images, mnist.test.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.

	return test

