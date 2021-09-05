import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

# Known parameters
M = 3
N = 50
sigma_noise = 0.1
epsilon = np.random.normal(0, sigma_noise, N)

# Loop parameters
epochs = 400
learning_rate = 0.03

# Noisy sin wave
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + epsilon

# Parameters to estimate
w = []
mu = []
sigma = []
b = []
for _ in range(M):
    w.append(tf.Variable(np.random.rand()))
    mu.append(tf.Variable(np.random.rand()))
    sigma.append(tf.Variable(np.random.rand()))
    b.append(tf.Variable(np.random.rand()))

# Gaussian 
def phi(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / sigma**2)

def y_hat(x):
    y_hat_i = 0
    for j in range(M):
        y_hat_i = y_hat_i + w[j]*phi(x, mu[j], sigma[j]) + b[j]
    return y_hat_i
     
def J(y, y_hat):
    return (1/2)*tf.square(y - y_hat)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(J(y, y_hat(x)))

    gradients = tape.gradient(loss, [w, mu, sigma, b])
    for j in range(M):
        w[j].assign_sub(gradients[0][j]*learning_rate)
        mu[j].assign_sub(gradients[1][j]*learning_rate)
        sigma[j].assign_sub(gradients[2][j]*learning_rate)
        b[j].assign_sub(gradients[3][j]*learning_rate)

    print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")

plt.figure()
plt.title("")
plt.scatter(x, y, label="Noisy Sin Curve", color="green")
plt.plot(x, np.sin(2*np.pi*x), label="Sin Curve", color="red")
plt.plot(x, y_hat(x), label="Approximation", color="blue")
plt.legend()
plt.show()

'''
References:
(1) https://www.machinelearningplus.com/deep-learning/linear-regression-tensorflow/

'''


