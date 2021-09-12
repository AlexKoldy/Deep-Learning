'''
Alexander Koldy
ECE 472 - Deep Learning
Assignment 2
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.close('all')


class Spiral():
	def __init__(self, theta, direction=np.array([1, 1])):
		self.x = direction[0]*theta*np.cos(theta)
		self.y = direction[1]*theta*np.sin(theta)
	
	def add_noise(self, sigma_noise, N):
		epsilon = np.random.normal(0, sigma_noise, N)
		self.x += epsilon
		epsilon = np.random.normal(0, sigma_noise, N)
		self.y += epsilon

class Data():
	def __init__(self):
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

class Perceptron():
	def __init__(self, input, activation_type):
		'''Perceptron parameters'''
		self.W = np.random.uniform(-100, 100, (input.shape))
		self.b = np.random.uniform(-100, 100)

		'''Perceptron input'''
		self.input = input

		'''Establish activation type'''
		self.activation_type = activation_type

	def neuron(self):
		def activation(self, t):
			if self.activation_type == 'ReLu':
				return self.relu(t)
			elif self.activation_type == 'Sigmoid':
				return self.sigmoid(t)

		return self.activation(np.sum(self.W*self.input) + self.b)

	def relu(self, t):
		return np.maximum(0, t)
	
	def sigmoid(self, t):
		return 1 / (1 + np.exp(-t))

class Neural_Network():
	
	class Layer():
		def __init__(self, input, width, activation_type):
			self.perceptrons = []
			self.W = []
			self.b = []

			for _ in range(width):
				perceptron = Perceptron(self.input, activation_type)
				self.perceptrons.append(perceptron)
				self.W.append(perceptron.W)
				self.b.append(perceptron.b)
		
	'''
	Establish a neural network with: 
	X: (x, y) -> input layer
	depth: number of hidden layers
	width: list of size depth containing
	desired width of each layer
	'''
	def __init__(self, X, depth, width):
		self.X = X
		self.depth = depth
		self.layers = []
		self.W = []
		self.b = []
		
		self.create_layer(self.X, width[0], 'Sigmoid')
		previous_layer = self.layers[0]
		for _ in range(depth):
			self.create_layer()
			
	def create_layer(self, input, width, activation_type):
		layer = self.Layer(input, width, activation_type)
		self.layers.append(layer)
		self.W.append(layer.W)
		self.b.append(layer.b)
	
	def Y(self):
		pass

	




	def loss(self, y, y_hat):
		return tf.reduce_mean(-y*np.log(y_hat) - (1 - y)*np.log(1 - y_hat))

	
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
'''