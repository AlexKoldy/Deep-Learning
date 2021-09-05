# Import Relevant libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# Learning rate
learning_rate = 0.01

# Number of loops for training through all your data to update the parameters
training_epochs = 100

# the training dataset
x_train = np.linspace(0, 10, 100)
y_train = x_train + np.random.normal(0,1,100)

# plot of data
plt.scatter(x_train, y_train)

# declare weights
weight = tf.Variable(0.)
bias = tf.Variable(0.)

# Define linear regression expression y
def linreg(x):
    y = weight*x + bias
    return y

# Define loss function (MSE)
def squared_error(y_pred, y_true):
    return tf.reduce_mean((y_pred - y_true)**2)

# train model
for epoch in range(training_epochs):

    # Compute loss within Gradient Tape context
    with tf.GradientTape() as tape:
        y_predicted = linreg(x_train)
        loss = squared_error(y_predicted, y_train)

    # Get gradients
    gradients = tape.gradient(loss, [weight,bias])

    # Adjust weights
    weight.assign_sub(gradients[0]*learning_rate)
    bias.assign_sub(gradients[1]*learning_rate)

    # Print output
    print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")
