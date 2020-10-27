import tensorflow as tf
import numpy as np
import copy
from colorama import init
from termcolor import *
import cv2
from skimage.color import rgb2yiq, yiq2rgb
init()
from io import BytesIO
import numpy as np
import requests
import matplotlib.pyplot as plt

def init_param():
	url = 'https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz'
	r = requests.get(url, stream = True)
	data = np.load(BytesIO(r.raw.read()))
	return data

def hist_match(C, S):
	mean_C = np.mean(C)
	mean_S = np.mean(S)
	var_C = np.std(C)
	var_S = np.std(S)
	return var_C * ( S - mean_S)/var_S + mean_C


# VGG16 components
class Dense(tf.keras.layers.Layer):
	def __init__(self, data, param_name, f= lambda x: x, disp_name=True):
		super(Dense, self).__init__()
		self.W = tf.Variable(data[param_name+'_W'].astype('float64'), name=param_name+'_W', trainable=False)
		self.b = tf.Variable(data[param_name+'_b'].astype('float64'), name=param_name+'_b', trainable=False)
		self.f = f
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) + self.b)

class Flatten:
	def __init__(self, disp_name=True):
		self.name = 'Flatten'
		if disp_name == True:
			print('\tAdding',self.name)

	@staticmethod
	def forward(X):
		return tf.reshape(X, [tf.shape(X)[0], -1])

class MaxPoolLayer:
	def __init__(self, dim, stride, disp_name=True):
		self.dim = dim
		self.stride = stride
		self.name = 'MaxPoolLayer'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.max_pool(
			X,
			ksize=[1, self.dim, self.dim, 1],
			strides=[1, self.stride, self.stride, 1],
			padding='SAME')

class AvgPoolLayer:
	def __init__(self, dim, stride, disp_name=True):
		self.dim = dim
		self.stride = stride
		self.name = 'AvgPoolLayer'
		if disp_name == True:
			print('\tAdding',self.name)

	def forward(self, X):
		return tf.nn.avg_pool(
			X,
			ksize=[1, self.dim, self.dim, 1],
			strides=[1, self.stride, self.stride, 1],
			padding='SAME')

class ConvLayer(tf.keras.layers.Layer):
	def __init__(self, data, param_name,stride=1, padding='SAME', disp_name=True):
		super(ConvLayer, self).__init__()
		self.W = tf.Variable(data[param_name+'_W'].astype('float64'), name=param_name+'_W', trainable=False)
		self.b = tf.Variable(data[param_name+'_b'].astype('float64'), name=param_name+'_b', trainable=False)
		self.stride = stride
		self.padding = padding
		if disp_name==True:
			print('\tAdding',self.name)

	def forward(self, x):
		x = tf.nn.conv2d(x, self.W, strides=[1, self.stride, self.stride, 1], padding=self.padding)
		x = x + self.b
		return tf.nn.relu(x)

# VGG16 
class VGG16(tf.keras.layers.Layer):
	def __init__(self, data):
		super(VGG16, self).__init__()
		cprint('Initializing VGG16............', 'green')
		self.layers = [
		# ConvLayer1
		ConvLayer(data, param_name='conv1_1'),
		ConvLayer(data, param_name='conv1_2'),
		AvgPoolLayer(dim=2, stride=2),
		# ConvLayer2
		ConvLayer(data, param_name='conv2_1'),
		ConvLayer(data, param_name='conv2_2'),
		AvgPoolLayer(dim=2, stride=2),
		# ConvLayer3
		ConvLayer(data, param_name='conv3_1'),
		ConvLayer(data, param_name='conv3_2'),
		ConvLayer(data, param_name='conv3_2'),
		AvgPoolLayer(dim=2, stride=2),
		# ConvLayer4
		ConvLayer(data, param_name='conv4_1'),
		ConvLayer(data, param_name='conv4_2'),
		ConvLayer(data, param_name='conv4_2'),
		AvgPoolLayer(dim=2, stride=2),
		# ConvLayer5
		ConvLayer(data, param_name='conv5_1'),
		# ConvLayer(data, param_name='conv5_2'),
		# ConvLayer(data, param_name='conv5_2'),
		# AvgPoolLayer(dim=2, stride=2),
		# # Flatten
		# Flatten(),
		# # Dense
		# Dense(data, f=tf.nn.relu, param_name='fc6'),
		# Dense(data, f=tf.nn.relu, param_name='fc7'),
		# Dense(data, f=tf.nn.softmax, param_name='fc8')
		]
		cprint('Initializing Complete!!!............', 'green')

	def forward(self, X):
		i = 0
		X_style = []
		X_content = []
		for layer in self.layers:
			X = layer.forward(X)
			if i in [0, 3, 6, 10, 14]:
				X_style.append(X)
			if i == 11:
				X_content.append(X)
			if i == 14:
				break			
			i += 1
		return X_style, X_content

class NSTModel(tf.keras.layers.Layer):
	def __init__(self, model):
		super(NSTModel, self).__init__()
		self.model = model
		self.X = tf.Variable(np.random.random((1,224,224,3)))

	def forward(self, X, C, S):
		Y_xs, Y_xc = self.model.forward(X)
		Y_s, _ = self.model.forward(S)
		_, Y_c = self.model.forward(C)
		return Y_xc, Y_xs, Y_c, Y_s

	def clip_0_1(self, X):
	  return tf.clip_by_value(X, clip_value_min=0.0, clip_value_max=1.0)

	def content_loss(self, X, C):
		for x,c in zip(X, C):
			out = 0.5*tf.norm(x-c, ord=2)**2

		return out

	def style_loss(self,weight_s, X, S):
		out = 0
		for x, s, w in zip(X, S,weight_s):
			_, x_h, x_w, _ = tf.shape(x)
			G_x = tf.linalg.einsum('bijc, bijd->bcd', x, x)/ (tf.cast(x_h, tf.float64)*tf.cast(x_w, tf.float64))
			
			_, s_h, s_w, _ = tf.shape(s)
			G_s = tf.linalg.einsum('bijc, bijd->bcd', s, s)/ (tf.cast(s_h, tf.float64)*tf.cast(s_w, tf.float64))

			out += w * 0.25 * tf.norm(G_x-G_s, ord='fro', axis=(1,2))**2
		
		return out

	def total_loss(self, Y, weight_s, alpha, beta):
		out = alpha * self.content_loss(Y[0], Y[2]) + beta * self.style_loss(weight_s, Y[1], Y[3])
		
		return out

	def train(self, X, C, S, weight_s, alpha=1e-2, beta=1, lr=0.8):
		self.X.assign(X)
		optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
		for epoch in range(200):
			with tf.GradientTape() as t:
				t.watch(self.X)
				Y = self.forward(self.X, C, S)
				Loss = self.total_loss(Y, weight_s, alpha, beta)

			grads = t.gradient(Loss, self.X)
			cprint(f'Epoch: {epoch}, Loss: {Loss}','green')	

			optimizer.apply_gradients(zip([grads], [self.X]))
			self.X.assign(self.clip_0_1(self.X))
			I = self.X.numpy().reshape(224,224,3)
			# plt.imshow(yiq2rgb(I))
			# plt.show()

		return self.X, Loss

if __name__ == '__main__':
	# Content Image
	C = cv2.imread('Taj-Mahal.jpg')
	# C = cv2.imread('Content/Taj-Mahal.jpg')
	C = cv2.resize(C, (224, 224))/ 255
	C = rgb2yiq(C)
	

	# Style Image
	S = cv2.imread('seated_nude.jpg')
	# S = cv2.imread('Style/seated_nude.jpg')
	S = cv2.resize(S, (224, 224)) / 255
	S = rgb2yiq(S)
	

	# Luminance Histogram Matching
	S[:,:,0] = hist_match(C[:,:,0], S[:,:,0]) 

	Cp = np.tensordot( C[:,:,0], np.ones((3)),axes=0)
	Sp = np.tensordot( S[:,:,0], np.ones((3)),axes=0)

	Sp = np.expand_dims(S, axis=0)
	Cp = np.expand_dims(C, axis=0) 


	# Initiliatize Model
	params = init_param()
	vgg = VGG16(params)
	model = NSTModel(vgg)
	X = tf.Variable(np.random.random((1,224,224,3)).astype('float64'), name='GenImg', trainable=True)
	Lt, Loss = model.train(X, Cp, Sp, weight_s=[3/64, 3/128, 3/256, 3/512, 3/512])
	
	I = Lt.numpy().reshape(224,224,3)
	I[:,:,1] = C[:,:,1]
	I[:,:,2] = C[:,:,2]
	I = yiq2rgb(I)
	C = yiq2rgb(tf.reshape(C,[224,224,3]).numpy())
	S = yiq2rgb(tf.reshape(S,[224,224,3]).numpy())

	plt.figure()
	plt.subplot(131)
	plt.imshow(C)
	plt.title('Context')
	plt.subplot(132)
	plt.imshow(S)
	plt.title('Style')
	plt.subplot(133)
	plt.imshow(I)
	plt.title('NST')
	plt.savefig('comparision.jpg')
	plt.show()
	cv2.imwrite('NST_Image.jpg', I*255)
