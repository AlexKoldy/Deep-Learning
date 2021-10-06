'''
Alexander Koldy
ECE 472 - Deep Learning
Assignment 4: Classify CIFAR10. Acheive performance similar 
to the state of the art. ClassifyCIFAR100. Achieve a top-5 
accuracy of 90%
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Data():
	def __init__(self, files, is_training):
		'''Training, validation, and testing data parameters'''
		self.image_shape = (32, 32)
		self.num_training = 40000
		self.num_validation = 10000
		self.num_testing = 10000

		self.load_data(files, is_training)

	def load_data(self, files, is_training):		
		first_file = True
		for file in files:
			dict = self.unpickle(file)

			if is_training:
				if first_file == True:
					self.training_data = dict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
					self.training_labels = np.array(dict['labels'])
					first_file = False	
				else:
					self.training_data = np.concatenate((self.training_data, dict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")), axis=0)
					self.training_labels = np.concatenate((self.training_labels, np.array(dict['labels'])), axis=0)
			else:
				self.testing_data = dict['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
				self.testing_labels = np.array(dict['labels'])

		if is_training:		
			'''Establish training and validation data'''
			self.validation_data = self.training_data[self.num_training:, :, :, :]
			self.training_data = self.training_data[:self.num_training, :, :, :]
			#new_order = np.random.permutation(self.num_training)
			#self.training_data = self.training_data[new_order, :, :, :]


			'''Establish training and validation labels'''
			self.validation_labels = self.training_labels[self.num_training:].flatten()
			self.training_labels = self.training_labels[:self.num_training].flatten()
			#self.training_labels = self.training_labels[new_order]
			
			'''Normalize'''
			#self.training_data = self.training_data.astype('float32') / 255.0
			self.training_data = self.training_data / 255.0
			#self.validation_data = self.validation_data.astype('float32') / 255.0
			self.validation_data = self.validation_data / 255.0

			width_shift = 3/32
			height_shift = 3/32
			flip = True

			self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=flip, width_shift_range=width_shift, height_shift_range=height_shift)
			self.datagen.fit(self.training_data)
			
			'''
			def visualize_data(images):
				fig = plt.figure(figsize=(14, 6))
				fig.patch.set_facecolor('white')
				for i in range(3 * 7):
					plt.subplot(3, 7, i+1)
					plt.xticks([])
					plt.yticks([])
					plt.imshow(images[i])
				plt.show()
			

			it = self.datagen.flow(self.training_data, self.training_labels, shuffle=False)
			batch_images, batch_labels = next(it)
			visualize_data(batch_images)
			'''	

			
			
		else:
			#self.testing_data = self.testing_data.astype('float32') / 255.0
			self.testing_data = self.testing_data / 255.0

		def verify_load():
			print(self.validation_labels[9999])
			im_r = self.validation_data[9999, :, :, 0]
			im_g = self.validation_data[9999, :, :, 1]
			im_b = self.validation_data[9999, :, :, 2]
			img = np.dstack((im_r, im_g, im_b))
			plt.figure(0, figsize=(1,1))
			plt.imshow(img) 
			plt.show()

		
		
	def unpickle(self, file):
		import pickle
		with open(file, 'rb') as f:
			dict = pickle.load(f, encoding='latin1')
		return dict

class ResNet():
	def __init__(self):
		self.num_classes = 10
		self.input = tf.keras.Input(shape=(32, 32, 3))
		
		
	
		x =  tf.keras.layers.ZeroPadding2D(padding=(3, 3))(self.input)
		x = self.convolution_block(x, 1, 32)
		x = self.identity_block(x, 32)
		#x = tf.keras.layers.MaxPool2D((2, 2))

		x = self.convolution_block(x, 1, 64)
		x = self.identity_block(x, 64)
		#x = tf.keras.layers.MaxPool2D((2, 2))

		x = self.convolution_block(x, 1, 128)
		x = self.identity_block(x, 128)
		#x = tf.keras.layers.MaxPool2D((2, 2))

		

		'''
		x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
		x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

		x = self.convolution_block(x, 1, 64)
		x = self.identity_block(x, 64)
		x = self.identity_block(x, 64)

		x = self.convolution_block(x, 2, 128)
		x = self.identity_block(x, 128)
		x = self.identity_block(x, 128)
		x = self.identity_block(x, 128)
		
		x = self.convolution_block(x, 2, 256)
		x = self.identity_block(x, 256)
		x = self.identity_block(x, 256)
		x = self.identity_block(x, 256)
		x = self.identity_block(x, 256)
		x = self.identity_block(x, 256)
		
		x = self.convolution_block(x, 2, 512)
		x = self.identity_block(x, 512)
		x = self.identity_block(x, 512)
		'''
		
		x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(128, activation='relu')(x)
		self.output = tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer='he_normal')(x)

		self.model = tf.keras.Model(inputs=self.input, outputs=self.output)

	def __call__(self):
		return self.model

	def identity_block(self, x, filter):
		x_skip = x
		weight_decay = 0.001

		x = tf.keras.layers.Conv2D(filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)

		x = tf.keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
		
		x = tf.keras.layers.Conv2D(4*filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
		x = tf.keras.layers.BatchNormalization()(x)

		x = tf.keras.layers.Add()([x, x_skip])
		x = tf.keras.layers.Activation('relu')(x)

		return x
	
	def convolution_block(self, x, stride, filter):
		x_skip = x
		weight_decay = 0.001

		x = tf.keras.layers.Conv2D(filter, kernel_size=(1, 1), strides=(stride, stride), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)

		x = tf.keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)

		x = tf.keras.layers.Conv2D(4*filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
		x = tf.keras.layers.BatchNormalization()(x)

		x_skip = tf.keras.layers.Conv2D(4*filter, kernel_size=(1, 1), strides=(stride, stride), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x_skip)
		x_skip = tf.keras.layers.BatchNormalization()(x_skip)

		x = tf.keras.layers.Add()([x, x_skip])
		x = tf.keras.layers.Activation('relu')(x)

		return x




training_files = ['data/cifar-10-python/cifar-10-batches-py/data_batch_1', 
		 'data/cifar-10-python/cifar-10-batches-py/data_batch_2',
		 'data/cifar-10-python/cifar-10-batches-py/data_batch_3',
		 'data/cifar-10-python/cifar-10-batches-py/data_batch_4',
		 'data/cifar-10-python/cifar-10-batches-py/data_batch_5']
data = Data(training_files, True)

'''Establish Convolutional Neural Network'''
'''
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01),  input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(10, activation='softmax'))


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
'''

resnet = ResNet()
model = resnet()

model.summary()

'''Training/Validation'''
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_top_k_categorical_accuracy'])

print("Training/Validation Data!")
history = model.fit(data.datagen.flow(data.training_data, data.training_labels, batch_size=64), epochs=200, validation_data=(data.validation_data, data.validation_labels))

'''Testing'''
testing_files = ['data/cifar-10-python/cifar-10-batches-py/test_batch']
data = Data(testing_files, False)

print("Testing Data!")
history = model.evaluate(data.testing_data, data.testing_labels)

'''
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPool2D((2,2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(tf.keras.layers.MaxPool2D((2,2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(tf.keras.layers.MaxPool2D((2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


batch_size = 32
epochs = 50
m_no_aug = create_model()
m_no_aug.summary()

history_no_aug = m_no_aug.fit(
    data.training_data, data.training_labels,
    epochs=epochs, batch_size=batch_size,
    validation_data=(data.validation_data, data.validation_labels))

data = Data(testing_files, False)
m_no_aug.evaluate(data.testing_data,  data.testing_labels)
'''


'''
References:
(1) Cooper Union ECE-472: Deep Learning - Learning Materials
(2) https://www.cs.toronto.edu/~kriz/cifar.html
(3) https://stackoverflow.com/questions/35995999/why-cifar-10-images-are-not-displayed-properly-using-matplotlib
(4) https://arxiv.org/pdf/1409.1556v6.pdf
(5) https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
(6) https://stackoverflow.com/questions/52507306/keras-top-k-categorical-accuracy-metric-is-extremely-low-compared-to-accuracy?rq=1
(7) https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691
(8) https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
(9) https://stepup.ai/train_data_augmentation_keras/
'''