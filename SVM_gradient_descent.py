# !/usr/bin/python3
# Filename: SVM_gradient_descent.py
# Author: Marion Chiariglione
# Date Created: February 17th 2020
# Role: Implementation of the gradient descent to learn the SVM model

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

## DATA MANIPULATION ##

# Loads dataset into multidimensional array without the header information
def load_dataset(inputFile, nbRowsToSkip):
    data = np.loadtxt(inputFile,skiprows=nbRowsToSkip)
    return data

## GRADIENT DESCENT METHODS ##

# Calculates the change in the cost to check it against the convergence criteria / computed at the end of each iteration
def cost_change_calculation(prev_cost, current_cost):
    return (abs(prev_cost - current_cost) * 100) / prev_cost

# Calculates the error part of the loss function
def error(x_vec, y_vec, w_vec, b):
    return 1 - y_vec * (np.dot(x_vec, w_vec) + b)

# Calculates the loss function of the SVM
def loss_function(x_vec, y_vec, w_vec, b, C):
    return 0.5 * np.dot(w_vec, w_vec) + C * np.sum(np.maximum(np.zeros(len(y_vec)), error(x_vec, y_vec, w_vec, b)))

# Calculates the condition for the gradient, returns 0 or 1 or a vector of 0 and 1
def gradient_condition(x_vec, y_vec, w_vec, b):
    return (y_vec * (np.dot(x_vec, w_vec) + b) < 1).astype(int)

## BATCH ##

# Calculates the gradient for the batch gradient descent with regards to w
def calculate_gradient_batch(x_vec, y_vec, w_vec, b, C):
    part_gradient = np.zeros(len(w_vec))
    condition = gradient_condition(x_vec, y_vec, w_vec, b)
    for j in range(len(w_vec)):
        part_gradient[j] = np.sum(condition * (-y_vec*x_vec[:,j]))
    return w_vec + C * part_gradient

# Calculates the gradient for the batch gradient descent with regards to b
def calculate_gradient_b_batch(x_vec, y_vec, w_vec, b, C):
    condition = gradient_condition(x_vec, y_vec, w_vec, b)
    part_gradient = np.sum(condition * (-y_vec))
    return C * part_gradient

# ALgorithm of the batch gradient descent, returns the loss over the number of iterations
def batch_gradient_descent(x_vec, y_vec, w_vec, b, C, epsilon_batch, eta_batch):
    all_loss = []
    criteria = 1
    k = 0
    while criteria >= epsilon_batch:
        gradient = calculate_gradient_batch(x_vec, y_vec, w_vec, b, C)
        part_b = calculate_gradient_b_batch(x_vec, y_vec, w_vec, b, C)
        w_vec = w_vec - (eta_batch * gradient)
        b = b - (eta_batch * part_b)
        loss = loss_function(x_vec, y_vec, w_vec, b, C)
        all_loss.append(loss)
        if k == 0:
            criteria = 1
        else:
            criteria = cost_change_calculation(all_loss[k-1], all_loss[k])
        k += 1

    return all_loss, k

## STOCHASTIC ##

# Calculates the gradient for the stochastic gradient descent with regards to w
def calculate_gradient_stochastic(x_vec, y_vec, w_vec, b, C, i):
    part_gradient = np.zeros(len(w_vec))
    condition = gradient_condition(x_vec[i,:], y_vec[i], w_vec, b)
    for j in range(len(w_vec)):
        part_gradient[j] = condition * (-y_vec[i]*x_vec[i,j])
    return w_vec + C * part_gradient

# Calculates the gradient for the stochastic gradient descent with regards to b
def calculate_gradient_b_stochastic(x_vec, y_vec, w_vec, b, C, i):
    condition = gradient_condition(x_vec[i,:], y_vec[i], w_vec, b)
    part_b_gradient = condition * (-y_vec[i])
    return C * part_b_gradient

# Algorithm of the stochastic gradient descent, returns the loss over the number of iterations
def stochastic_gradient_descent(x_vec, y_vec, w_vec, b, C, epsilon_stochastic, eta_stochastic):
    all_loss = []
    criteria = 1
    delta_cost = 0
    k = 0
    i = 0
    crit = []
    while criteria >= epsilon_stochastic:
        gradient = calculate_gradient_stochastic(x_vec, y_vec, w_vec, b, C, i)
        part_b = calculate_gradient_b_stochastic(x_vec, y_vec, w_vec, b, C, i)
        w_vec = w_vec - (eta_stochastic * gradient)
        b = b - (eta_stochastic * part_b)
        loss = loss_function(x_vec, y_vec, w_vec, b, C)
        all_loss.append(loss)
        if k == 0:
            delta_cost = 1
            crit.append(0)
        else:
            delta_cost = cost_change_calculation(all_loss[-2], all_loss[-1])

        criteria = 0.5*crit[-1] + 0.5*delta_cost
        crit.append(criteria)

        k += 1
        i = (i + 1) % len(y_vec)

    return all_loss, k

## MINI BATCH ##

# Calculates the gradient for the mini batch gradient descent with regards to W
def calculate_gradient_miniB(x_vec, y_vec, w_vec, b, C):
    part_gradient = np.zeros(len(w_vec))
    condition = gradient_condition(x_vec, y_vec, w_vec, b)
    for j in range(len(w_vec)):
        part_gradient[j] = np.sum(condition * (-y_vec*x_vec[:,j]))
    return w_vec + C * part_gradient

# Calculates the gradient for the mini batch gradient descent with regards to b
def calculate_gradient_b_miniB(x_vec, y_vec, w_vec, b, C):
    condition = gradient_condition(x_vec, y_vec, w_vec, b)
    part_gradient_b = np.sum(condition * (-y_vec))
    return C * part_gradient_b

# Algorithm of the mini batch gradient descent, returns the loss over the number of iterations
def mini_batch_gradient_descent(x_vec, y_vec, w_vec, b, C, epsilon_miniB, eta_miniB, batch_size):
    all_loss = []
    criteria = 1
    k = 0
    s = 0
    e = batch_size
    crit = []
    i = 0

    while criteria >= epsilon_miniB:
        X_curr_batch = x_vec[i:i+batch_size, :]
        Y_curr_batch = y_vec[i:i+batch_size]

        gradient = calculate_gradient_miniB(X_curr_batch, Y_curr_batch, w_vec, b, C)
        part_b = calculate_gradient_b_miniB(X_curr_batch, Y_curr_batch, w_vec, b, C)
        w_vec = w_vec - (eta_miniB * gradient)
        b = b - (eta_miniB * part_b)
        loss = loss_function(x_vec, y_vec, w_vec, b, C)
        all_loss.append(loss)

        if k == 0:
            delta_cost = 1
            crit.append(0)
        else:
            delta_cost = cost_change_calculation(all_loss[-2], all_loss[-1])

        criteria = 0.5*crit[-1] + 0.5*delta_cost
        crit.append(criteria)

        k += 1
        i = (i + batch_size) % len(y_vec)

    return all_loss, k

## MAIN ##

inputFile = sys.argv[1]
data = load_dataset(inputFile, 13)

x_vec = data[:, :-1] # all columns but the last one = inputs / features
y_vec = data[:, -1] # last column = outputs / labels
w_vec = np.zeros(data.shape[1]-1) # weights
b = 0 # bias
C = 10 # penalty constant


epsilon_batch = 0.04 # convergence criteria
epsilon_stochastic = 0.0003
epsilon_miniB = 0.004

eta_batch = 0.000000001 # learning rate
eta_stochastic = 0.00000001
eta_miniB = 0.00000001

batch_size = 4

start1 = time.time()
loss1, k1 = batch_gradient_descent(x_vec, y_vec, w_vec, b, C, epsilon_batch, eta_batch)
end1 = time.time()
print("Time batch gradient descent (in sec): "+str(end1 - start1))

start2 = time.time()
loss2, k2 = stochastic_gradient_descent(x_vec, y_vec, w_vec, b, C, epsilon_stochastic, eta_stochastic)
end2 = time.time()
print("Time stochastic gradient descent (in sec): "+str(end2 - start2))

start3 = time.time()
loss3, k3 = mini_batch_gradient_descent(x_vec, y_vec, w_vec, b, C, epsilon_miniB, eta_miniB, batch_size)
end3 = time.time()
print("Time mini gradient descent (in sec): "+str(end3 - start3))

plt.subplots(figsize=(12,8))
plt.subplot(3,1,1)
plt.title('Error vs. Training Epoch')
plt.plot(np.arange(k1), loss1, 'b', label="Batch")
plt.legend(loc='upper right', frameon=False)
plt.ylabel('Cost')

plt.subplot(3,1,2)
plt.plot(np.arange(k2), loss2, 'r', label="Stochastic")
plt.legend(loc='upper right', frameon=False)
plt.ylabel('Cost')

plt.subplot(3,1,3)
plt.plot(np.arange(k3), loss3, 'g', label="Mini-Batch")
plt.legend(loc='upper right', frameon=False)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
