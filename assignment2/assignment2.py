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
		self.batch_size = 10

		'''Data generation parameters'''
		self.N = 200
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
	def get_batch(self, batch_size ):
		batch_start = np.random.randint(0, 2*self.N - batch_size - 1)
		batch_end = batch_start + batch_size
		X_batch = tf.convert_to_tensor(self.X[:, batch_start:batch_end].T)
		Y_batch = tf.convert_to_tensor(self.Y[batch_start:batch_end])

		return X_batch, Y_batch
	
	def convert_data(self):
		self.X = tf.convert_to_tensor(self.X.T)
		self.Y = tf.convert_to_tensor(self.Y)

'''
Multi-layer perceptron which contains 2
nested classes: Perceptron and Layer.
Constructs neural network to solve
binary classification problem.
'''
class Neural_Network(tf.Module):
	'''
	Layer of perceptrons which can
	be generated more than once
	(i.e., multiple hidden layers)
	'''
	class Layer(tf.Module):
		def __init__(self, activation_type, num_inputs, width):
			self.activation_type = activation_type
			self.width = width

			'''Perceptron parameters'''
			self.W = tf.Variable(tf.random.normal(stddev=0.001, shape=(num_inputs, width)))
			self.b = tf.Variable(tf.random.normal(stddev=0.1, shape=(1, width)))
			
		def update(self, input):
			def activation(t):
				if self.activation_type == 'ReLu':
					return tf.nn.relu(t)
				elif self.activation_type == 'Sigmoid':
					return tf.nn.sigmoid(t)

			return activation(input @ self.W + self.b)
				
	'''
	Establish a neural network with: 
	depth: number of hidden layers
	width: list of size depth containing
	desired width of each layer
	data: all training data available
	'''
	def __init__(self, depth, widths, data):
		'''Data'''
		self.data = data

		'''Learning parameters'''
		self.epochs = 1000
		self.learning_rate = 0.005

		'''Create neural network'''
		self.layers = []
		self.create_layer('ReLu', 2, widths[0])
		for i in range(1, depth - 1):
			self.create_layer('ReLu', widths[i - 1], widths[i])
		self.create_layer('Sigmoid', self.layers[-1].width, 1)
			
	def create_layer(self, activation_type, num_inputs, width):
		layer = self.Layer(activation_type, num_inputs, width)
		self.layers.append(layer)
	
	def loss(self, y, y_hat):
		return tf.reduce_mean(-y*tf.math.log(y_hat) - (1 - y)*tf.math.log(1 - y_hat))

	def train(self):
		num_batches = 100
		X, Y = self.data.get_batch(self.data.batch_size)

		for j in range(num_batches):
			optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

			for epoch in range(self.epochs):
				with tf.GradientTape() as tape:
					y_hat = []
					for i in range(self.data.batch_size):
						input = np.reshape(X[i, :], (1, 2))
						for layer in self.layers:
							input = layer.update(input)
						output = input
						y_hat.append(output)

					y_hat = tf.reshape(tf.convert_to_tensor(y_hat, dtype=tf.float64), (self.data.batch_size, ))

					loss = self.loss(Y, y_hat)
					
				gradients = tape.gradient(loss, self.trainable_variables)

				print(f"Batch #{j} & Epoch count {epoch}: Loss value: {loss.numpy()}")

				optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		
	def test(self, data):
		num_correct = 0
		num_incorrect = 0

		data.convert_data()
		X, Y = data.X, data.Y

		for i in range(2*data.N):
			input = np.reshape(X[i, :], (1, 2))
			for layer in self.layers:
				input = layer.update(input)
			output = input
			y_hat = np.round(output.numpy()[0][0])
			y = Y[i].numpy()

			print("y: " + str(y) + " | y_hat: " + str(y_hat))

			if y_hat == y:
				num_correct += 1
			else:
				num_incorrect += 1
		
		print("Number correct: " + str(num_correct))
		print("Number incorrect: " + str(num_incorrect))

		return num_incorrect
		
data = Data()				
nn = Neural_Network(2, [2000, 2000], data)

'''Shuffle data before starting'''
nn.data.shuffle()

good_prediction = False
while good_prediction == False:
	nn.train()
	if nn.test(data) == 0:
		good_prediction = True
				
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