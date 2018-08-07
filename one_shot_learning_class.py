#https://arxiv.org/pdf/1606.04080.pdf ## Matching Networks for One Shot Learning

import tensorflow as tf #version 1.4
import numpy as np
import get_S_B_MNIST as sb

class one_shot:

	def __init__(self, sess, S, K, train_rate): 
		self.train_rate = train_rate
		self.channel = 1  #MNIST
		self.output = 10
		self.height = 28
		self.width = 28
		self.cell_num = 512
		self.K = K

		with tf.name_scope("placeholder"):
			#input
			self.X = tf.placeholder(tf.float32, [None, self.height * self.width])
			self.reshapeX = tf.reshape(self.X, [-1, self.height, self.width, self.channel])
			
			#target
			self.Y = tf.placeholder(tf.float32, [None, self.output])
			
			#support set
			self.S_x = np.reshape(S[:, :28*28], [-1, self.height, self.width, self.channel])
			self.S_x = tf.constant(self.S_x, tf.float32) # [# S, 28, 28, 1]
			self.S_y = np.reshape(S[:, 28*28:], [-1, self.output])
			self.S_y = tf.constant(self.S_y, tf.float32) # [# S, 10]
			
			#batch_norm
			self.is_train = tf.placeholder(tf.bool)



		with tf.name_scope("conv_net_f_g"):
			self.f_prime = self.conv_net('f_prime', self.reshapeX) #batch, cell_num # -1 ~ +1
			self.g_prime = self.conv_net('g_prime', self.S_x) #support_S, cell_num # -1 ~ +1
		
		
		with tf.name_scope("TheFullyConditionalEmbedding_g"):
			self.g = self.TheFullyConditionalEmbedding_g(self.g_prime) #[support_S 개수, cell_num]
			
		
		with tf.name_scope("TheFullyConditionalEmbedding_f"):
			self.f = self.TheFullyConditionalEmbedding_f(self.f_prime, self.g, self.K) #[support_S 개수, cell_num]


		with tf.name_scope("cosine_similarity"):
			self.similarity = self.cosine_similarity(self.f, self.g) #[batch, # S]
		

		with tf.name_scope("y_hat"):
			#similarity softmax하고 true label이랑 곱하고 합하면 yhat나옴.
			self.similarity_softmax = tf.nn.softmax(self.similarity, 1) #row별로 합이 1이 되도록 softmax 처리됨. [f_prime_batch, S 개수]
			self.y_hat = tf.matmul(self.similarity_softmax, self.S_y) #이것도 row별로 합이 1이 됨. support set이라서


		with tf.name_scope("train"):
			self.cost = -tf.reduce_mean( tf.reduce_sum( self.Y * tf.log( tf.clip_by_value(self.y_hat, 1e-15, 1.0) ), axis=1 ) ) 

			#Batch norm 학습 방법 : https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/layers/batch_normalization
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.minimize = tf.train.AdamOptimizer(self.train_rate).minimize(self.cost)

			self.correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(self.Y, 1)), tf.int32))

		sess.run(tf.global_variables_initializer())


	def conv_net(self, scope, data): #논문에서는 vgg or inception 쓰라고 되어 있음, MNIST는 사이즈가 작아서 작은 conv_net 으로 진행.
	#return shape: batch, self.cell_num
		with tf.variable_scope(scope): #scope == 'g' or 'f' #data == [batch, 28, 28, 1]	
			activation = tf.nn.relu

			layer1 = tf.layers.conv2d(data, filters=256, kernel_size = [3,3], strides=[1, 1], padding='SAME') 
			bn_layer1 = tf.layers.batch_normalization(layer1, training=self.is_train)
			layer1 = activation(bn_layer1)	
			pool_layer1 = tf.layers.max_pooling2d(layer1, pool_size = [2, 2], strides=[2, 2], padding='SAME') 

			layer2 = tf.layers.conv2d(pool_layer1, filters=512, kernel_size = [3,3], strides=[1, 1], padding='SAME') 
			bn_layer2 = tf.layers.batch_normalization(layer2, training=self.is_train)
			layer2 = activation(bn_layer2)
			pool_layer2 = tf.layers.max_pooling2d(layer2, pool_size = [2, 2], strides=[2, 2], padding='SAME')
			
			layer3 = tf.layers.conv2d(pool_layer2, filters=1024, kernel_size = [3,3], strides=[1, 1], padding='SAME') 
			bn_layer3 = tf.layers.batch_normalization(layer3, training=self.is_train)
			layer3 = activation(bn_layer3)
			pool_layer3 = tf.layers.max_pooling2d(layer3, pool_size = [2, 2], strides=[2, 2], padding='SAME') 

			flat = tf.layers.flatten(pool_layer3) # batch, 4*4*256
	
			FC = tf.layers.dense(flat, units = 2048)#, activation=None) # batch, self.cell_num
			bn_FC = tf.layers.batch_normalization(FC, training=self.is_train)
			FC = activation(bn_FC)
			
			output = tf.layers.dense(FC, units = self.cell_num)#, activation=None) # batch, self.cell_num
			bn_output = tf.layers.batch_normalization(output, training=self.is_train)
			output = tf.nn.tanh(bn_output)
			return output
			

	def TheFullyConditionalEmbedding_g(self, g_prime):
		#g_prime shape: [Support_S 개수, cell_num]
		fw = tf.nn.rnn_cell.LSTMCell(self.cell_num) # cell must write here. not in def function
		bw = tf.nn.rnn_cell.LSTMCell(self.cell_num)
	
		val, state = tf.nn.bidirectional_dynamic_rnn(fw, bw, tf.expand_dims(g_prime, 0), dtype=tf.float32)
		fw_val, bw_val = val[0], val[1] # [1, support_S 개수, cell_num]
		
		return (fw_val[0] + bw_val[0] + g_prime)/3 # -3~+3 => -1~+1  # [support_S 개수, cell_num]
		

	def TheFullyConditionalEmbedding_f(self, f_prime, g, K):
		#f_prime shape: [batch, cell_num]
		#g shape: [Support_S 개수, cell_num]
		#K: 반복수
		lstm = tf.nn.rnn_cell.LSTMCell(self.cell_num)

		expand_f_prime = tf.reshape(f_prime, [-1, 1, self.cell_num]) # [batch, 1, cell_num]
		transposed_g = tf.transpose(g, (1, 0)) # [cell_num, S 개수]
		
		_, state = tf.nn.dynamic_rnn(lstm, inputs=expand_f_prime, dtype=tf.float32)
		c, h = state.c, state.h #c = c_k-1 h = h_k-1

		#paper appendix A.1 
		for i in range(K):
			#(6)
			dot = tf.matmul(h, transposed_g) # 내적: [f_prime_batch, S 개수]
			dot_softmax = tf.nn.softmax(dot, 1) #row별로 합이 1이 되도록 softmax 처리됨. [f_prime_batch, S 개수]

			#(5)
			r = tf.matmul(dot_softmax, g) #shape: [f_prime_batch, cell_num]
						
			#(3) #논문에선 initial_state를 [h, r] 즉 concat 하라했는데 concat하면 shape가 안맞음. 따라서 NN으로 shape 맞춰줌.
			concat_h_r = tf.layers.dense(tf.concat((h,r), axis=1), self.cell_num)
			concat_h_r = tf.nn.tanh(concat_h_r)
			initial_state = tf.contrib.rnn.LSTMStateTuple(c = c, h = concat_h_r)
			
			_, state = tf.nn.dynamic_rnn(lstm, inputs=expand_f_prime, dtype=tf.float32, initial_state=initial_state)

			#(4)
			c = state.c ## c update
			h = (state.h + f_prime)/2 ## -2 ~ 2 => -1 ~ 1 h update
			
		return h

	
	def cosine_similarity(self, f, g): # (A 내적 B) / (A크기 * B크기) 	
		#f shape: [batch, cell_num]
		#g shape: [Support_S 개수, cell_num]
		transposed_g = tf.transpose(g, (1, 0)) # [cell_num, S 개수]
		dot = tf.matmul(f, transposed_g) # 내적: [batch, S 개수] #row당 S와의 내적 결과 가지고 있음.
		
		f_size = tf.sqrt(tf.reduce_sum(f*f, axis=1)) # batch
		g_size = tf.sqrt(tf.reduce_sum(g*g, axis=1)) # support_S 개수

		f_size = tf.reshape(f_size, [-1, 1])
		g_size = tf.reshape(g_size, [1, -1])
		mul_f_size_with_g_size = tf.matmul(f_size, g_size) #[batch, # S]

		similarity = dot/ (mul_f_size_with_g_size+1e-15)
		return similarity

