'''
Alexander Koldy
ECE 472 - Deep Learning
Assignment 2
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

'''
Generates noisy spirals for training data
'''
class Spiral():
	def __init__(self, theta, direction=np.array([1, 1])):
		self.x = direction[0]*theta*np.cos(theta)
		self.y = direction[1]*theta*np.sin(theta)
	
	def add_noise(self, sigma_noise, N):
		epsilon = np.random.normal(0, sigma_noise, N)
		self.x += epsilon
		epsilon = np.random.normal(0, sigma_noise, N)
		self.y += epsilon

'''
Collects data from noisy spirals.
Has shuffling functionality, and
is able to create batches for 
training
'''
class Data():
	def __init__(self):
		'''Data organization parameters'''
		self.batch_size = 50

		'''Data generation parameters'''
		self.N = 500
		self.sigma_noise = 0.2
		self.theta = np.linspace((1/4)*np.pi, 4.5*np.pi, self.N)
		self.direction_0 = np.array([-1, 1])
		self.direction_1 = np.array([1, -1])

		'''Generate Spirals'''
		self.spiral_0 = Spiral(self.theta, self.direction_0)
		self.spiral_0.add_noise(self.sigma_noise, self.N)
		self.spiral_1 = Spiral(self.theta, self.direction_1)
		self.spiral_1.add_noise(self.sigma_noise, self.N)
		spiral_x = np.concatenate((self.spiral_0.x, self.spiral_1.x), axis=0)
		spiral_y = np.concatenate((self.spiral_0.y, self.spiral_1.y), axis=0)

		'''Input/output data for training'''
		self.X = np.vstack((spiral_x, spiral_y))
		self.Y = np.concatenate((np.zeros((self.N, )), np.ones((self.N, ))), axis=0)
		
	def shuffle(self):
		new_order = np.random.permutation(2*self.N)
		self.X = self.X[:, new_order]
		self.Y = self.Y[new_order]

	# Does it matter if I use a point that was already used??
	def get_batch(self, batch_size=50):
		batch_start = np.random.randint(0, 2*self.N - batch_size - 1)
		batch_end = batch_start + batch_size
		X_batch = self.X[:, batch_start:batch_end]
		Y_batch = self.Y[batch_start:batch_end]

		return X_batch, Y_batch

'''
Multi-layer perceptron which contains 2
nested classes: Perceptron and Layer.
Constructs neural network to solve
binary classification problem.
'''
class Neural_Network():
	'''
	Perceptron which has [potentially]
	multiple inputs, and one output
	'''
	class Perceptron():
		def __init__(self, activation_type, input_shape):
			'''Perceptron parameters'''
			self.W = tf.Variable(np.random.uniform(-10, 10, (input_shape)))
			self.b = tf.Variable(np.random.uniform(-10, 10))

			'''Establish activation type'''
			self.activation_type = activation_type

		def neuron(self):
			def activation(t):
				if self.activation_type == 'ReLu':
					return self.relu(t)
				elif self.activation_type == 'Sigmoid':
					return self.sigmoid(t)

			return tf.convert_to_tensor(activation(np.sum(self.W.numpy()*self.input) + self.b))

		def relu(self, t):
			return np.math.maximum(0, t)
		
		def sigmoid(self, t):
			return 1 / (1 + tf.math.exp(-t))

		def update(self, input, W, b):
			'''Perceptron input'''
			self.input = input

			'''Perceptron parameters'''
			self.W = W
			self.b = b

	'''
	Layer of perceptrons which can
	be generated more than once
	(i.e., multiple hidden layers)
	'''
	class Layer():
		def __init__(self, width, activation_type, input_shape):
			self.width = width

			self.perceptrons = []
			self.W = []
			self.b = []

			for _ in range(width):
				perceptron = Neural_Network.Perceptron(activation_type, input_shape)
				self.perceptrons.append(perceptron)
				self.W.append(perceptron.W)
				self.b.append(perceptron.b)
		
		def update(self, input):
			self.output = np.zeros((len(self.perceptrons), 1))
			i = 0
			for perceptron in self.perceptrons:
				perceptron.update(input, perceptron.W, perceptron.b)
				self.output[i] = perceptron.neuron()
				i += 1
			
	'''
	Establish a neural network with: 
	depth: number of hidden layers
	width: list of size depth containing
	desired width of each layer
	data: all training data available
	'''
	def __init__(self, depth, width, data):
		'''Data'''
		self.data = data

		'''Learning parameters'''
		self.epochs = 1
		self.learning_rate = 0.1

		'''Track layers, weights, and biases'''
		self.layers = []
		self.W = []
		self.b = []
		
		'''Create neural network'''
		self.create_layer(width[0], 'Sigmoid', (2, 1))
		for i in range(depth - 1):
			self.create_layer(width[i + 1], 'ReLu', (self.layers[i - 1].width, 1))
		self.create_layer(1, 'Sigmoid', (self.layers[-1].width, 1))
		
		'''Parameters for optimization'''
		self.W = sum(self.W, [])
		self.b = sum(self.b, [])
			
	def create_layer(self, width, activation_type, input_shape):
		layer = self.Layer(width, activation_type, input_shape)
		self.layers.append(layer)
		self.W.append(layer.W)
		self.b.append(layer.b)
	
	def loss(self, y, y_hat):
		return tf.reduce_mean(-y*tf.math.log(y_hat) - (1 - y)*tf.math.log(1 - y_hat))

	def train(self):
		'''Shuffle data before starting'''
		self.data.shuffle()
		X, Y = self.data.get_batch()

		for epoch in range(self.epochs):
			with tf.GradientTape() as tape:
				tape.watch([self.W, self.b])
				y_hat = np.zeros(Y.shape)
				for i in range(self.data.batch_size):
					input = X[:, i]
					for layer in self.layers:
						layer.update(input)
						input = layer.output
					y_hat[i] = self.layers[-1].output
				y_hat = tf.convert_to_tensor(y_hat)
				Y = tf.convert_to_tensor(Y)
				
				print("Y: ")
				print(Y)
				print("y_hat: ")
				print(tf.convert_to_tensor(y_hat))

				loss = self.loss(Y, y_hat)

				print("loss: ")
				print(-Y*tf.math.log(y_hat) - (1 - Y)*tf.math.log(1 - y_hat))

				print("full_loss: ")
				print(tf.reduce_mean(-Y*tf.math.log(y_hat) - (1 - Y)*tf.math.log(1 - y_hat)))
			
			gradients = tape.gradient(loss, [self.W, self.b])
			print("gradient")
			print(gradients)
			


			
			
data = Data()				
nn = Neural_Network(1, [2], data)
nn.train()
				





	
def plot_results():
	data = Data()
	plt.figure(figsize=(10, 10))
	plt.scatter(data.spiral_1.x, data.spiral_1.y, color='red', edgecolors='black', s=15)
	plt.scatter(data.spiral_2.x, data.spiral_2.y, color='blue', edgecolors='black', s=15)
	plt.show()

		

'''
References:
(1) Cooper Union ECE-472: Deep Learning - Learning Materials
(2) https://www.codegrepper.com/code-examples/python/draw+spiral+in+matplotlib
(3) https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
(4) https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
'''