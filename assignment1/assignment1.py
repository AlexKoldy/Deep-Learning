import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

# Known parameters
M = 5
N = 50
sigma_noise = 0.1
epsilon = np.random.normal(0, sigma_noise, N)

# Loop parameters
epochs = 100
learning_rate = 0.01

# Noisy sin wave
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + epsilon

# Parameters to estimate
w = tf.Variable(0.0)
mu = tf.Variable(0.0)
sigma = tf.Variable(0.01)
b = tf.Variable(0.0)

# Gaussian 
def phi(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / sigma**2)

def y_hat(x):
    return tf.reduce_sum(w*phi(x, mu, sigma) + b)
     
def J(y, y_hat):
    return (1/2)*tf.square(y - y_hat)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(J(y, y_hat(x)))

    gradients = tape.gradient(loss, [w, mu, sigma, b])
    w.assign_sub(gradients[0]*learning_rate)
    #print(gradients)
    mu.assign_sub(gradients[1]*learning_rate)
    sigma.assign_sub(gradients[2]*learning_rate)
    b.assign_sub(gradients[3]*learning_rate)

    print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")



