'''
Alexander Koldy
ECE 472 - Deep Learning
Assignment 3: Classify MNIST digits with a 
(optionally convoultional) neural network. 
Get at least95.5% accuracy on the test test.
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gzip

class Data():
	def __init__(self):
		'''Training, validation, and testing data parameters'''
		self.image_shape = (28, 28)
		self.num_training = 50000
		self.num_validation = 10000
		self.num_testing = 10000

	def load_data(self, training_data_path, training_label_path, testing_data_path, testing_label_path):
		training_data = gzip.open(training_data_path, 'r')
		training_labels = gzip.open(training_label_path, 'r')
		testing_data = gzip.open(testing_data_path, 'r')
		testing_labels = gzip.open(testing_label_path, 'r')
		training_data.read(16)
		training_labels.read(8)
		testing_data.read(16)
		testing_labels.read(8)

		'''Establish training and validation data'''
		training_data = np.frombuffer(training_data.read(self.image_shape[0]*self.image_shape[1]*(self.num_training + self.num_testing)), dtype=np.uint8).astype(np.float64)
		training_data = training_data.reshape(self.num_training + self.num_validation, self.image_shape[0], self.image_shape[1], 1)
		self.validation_data = training_data[50000:, :, :, :]
		self.training_data = training_data[:50000, :, :, :]

		'''Establish training and validation labels'''
		training_labels = np.frombuffer(training_labels.read(self.num_training + self.num_validation), dtype=np.uint8).astype(np.float64)
		self.validation_labels = training_labels[50000:]
		self.training_labels = training_labels[:50000]

		'''Establish testing data'''
		testing_data = np.frombuffer(testing_data.read(self.image_shape[0]*self.image_shape[1]*self.num_testing), dtype=np.uint8).astype(np.float64)
		testing_data = testing_data.reshape(self.num_testing, self.image_shape[0], self.image_shape[1], 1)
		self.testing_data = testing_data

		'''Establish testing labels'''
		testing_labels = np.frombuffer(testing_labels.read(self.num_testing), dtype=np.uint8).astype(np.float64)
		self.testing_labels = testing_labels

		#print(self.validation_labels)
		#image = np.asarray(self.training_data[107]).squeeze()
		#plt.imshow(image)
		#plt.show()

'''Create data'''
training_data_path = 'data/train-images-idx3-ubyte.gz'
training_label_path = 'data/train-labels-idx1-ubyte.gz'
testing_data_path = 'data/t10k-images-idx3-ubyte.gz'
testing_label_path = 'data/t10k-labels-idx1-ubyte.gz'

data = Data()
data.load_data(training_data_path, training_label_path, testing_data_path, testing_label_path)

'''Establish Convolutional Neural Network'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=56, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=56, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(56, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

'''Training/Validation'''
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(data.training_data, data.training_labels, epochs=3, validation_data=(data.validation_data, data.validation_labels))

'''
References:
(1) Cooper Union ECE-472: Deep Learning - Learning Materials
(2) https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
(3) https://www.tensorflow.org/tutorials/images/cnn
'''


		



