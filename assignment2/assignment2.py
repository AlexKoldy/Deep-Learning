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
		X_batch = tf.convert_to_tensor(self.X[:, batch_start:batch_end])
		Y_batch = tf.convert_to_tensor(self.Y[batch_start:batch_end])

		return X_batch, Y_batch
	
	def convert_data(self):
		self.X = tf.convert_to_tensor(self.X)
		self.Y = tf.convert_to_tensor(self.Y)

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
			self.W = tf.Variable(np.random.uniform(-1, 1, (input_shape)))
			self.b = tf.Variable(np.random.uniform(-1, 1), dtype=tf.float64)

			'''Establish activation type'''
			self.activation_type = activation_type

		def neuron(self):
			def activation(t):
				if self.activation_type == 'ReLu':
					return tf.nn.relu(t)
				elif self.activation_type == 'Sigmoid':
					return tf.nn.sigmoid(t)

			return activation(tf.reduce_sum(self.W * self.input) + self.b)

		def update(self, input):
			'''Perceptron input'''
			self.input = input

		def step(self, W, b):
			'''Perceptron parameters'''
			self.W.assign_sub(W)
			self.b.assign_sub(b)

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

			self.input_shape = input_shape

			for _ in range(width):
				perceptron = Neural_Network.Perceptron(activation_type, input_shape)
				self.perceptrons.append(perceptron)
				self.W.append(perceptron.W)
				self.b.append(perceptron.b)
		
		def update(self, input):
			self.output = []

			for perceptron in self.perceptrons:
				perceptron.update(input)
				self.output.append(perceptron.neuron())

		def step(self, W, b, learning_rate):
			i = 0
			for perceptron in self.perceptrons:
				perceptron.step(W[i]*learning_rate, b[i]*learning_rate)
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
		self.epochs = 100
		self.learning_rate = 2

		'''Track layers, weights, and biases'''
		self.layers = []
		self.W = []
		self.b = []
	
		'''Create neural network'''
		self.create_layer(width[0], 'Sigmoid', (2, 1))
		for i in range(depth - 1):
			self.create_layer(width[i + 1], 'Sigmoid', (self.layers[i - 1].width, 1))
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
				y_hat = []
				for i in range(self.data.batch_size):
					input = np.reshape(X[:, i], (2, 1))
					for layer in self.layers:
						layer.update(input)
						input = layer.output
					y_hat.append(self.layers[-1].output[0])

				y_hat = tf.convert_to_tensor(y_hat)
				loss = self.loss(Y, y_hat)
			
			gradients = tape.gradient(loss, [self.W, self.b])

			print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")

			self.optimize(gradients)

	def optimize(self, gradients):
		start = 0
		end = 0
		for layer in self.layers:
			end += layer.width
			W = np.asarray(gradients)[0][start:end]
			b = np.asarray(gradients)[1][start:end]
			layer.step(W, b, self.learning_rate)
			start += layer.width
	
	def test(self, data):
		num_correct = 0
		num_incorrect = 0

		data.convert_data()
		X, Y = data.X, data.Y

		for i in range(2*data.N):
			input = np.reshape(X[:, i], (2, 1))
			for layer in self.layers:
				layer.update(input)
				input = layer.output
			y_hat = self.layers[-1].output[0].numpy()
			y = Y[i].numpy()

			print("y: " + str(y) + " | y_hat: " + str(y_hat))

			if (y_hat != 0 and y_hat != 1) or (y != 0 and y != 1):
				print("NO!")
			if y_hat == y:
				num_correct += 1
			else:
				num_incorrect += 1
		
		print("Number correct: " + str(num_correct))
		print("Number incorrect: " + str(num_incorrect))
		

			

		
data = Data()				
nn = Neural_Network(3, [16, 16, 16], data)
nn.train()
nn.test(data)
				





	
def plot_results():
	data = Data()
	plt.figure(figsize=(10, 10))
	plt.scatter(data.spiral_0.x, data.spiral_0.y, color='red', edgecolors='black', s=15)
	plt.scatter(data.spiral_1.x, data.spiral_1.y, color='blue', edgecolors='black', s=15)
	plt.show()

		

'''
References:
(1) Cooper Union ECE-472: Deep Learning - Learning Materials
(2) https://www.codegrepper.com/code-examples/python/draw+spiral+in+matplotlib
(3) https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
(4) https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
'''