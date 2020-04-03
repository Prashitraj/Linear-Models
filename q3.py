import numpy as np 
import matplotlib.pyplot as plt
import csv
import math

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(x, y, theta):
    h = sigmoid(np.dot(x, theta))
    cost = y * np.log(h) + (1 - y) * np.log(1 - h)
    return -cost.mean()

y = np.genfromtxt('../ass1_data/data/q3/logisticY.csv',delimiter=',')
x = np.ones((len(y),3))
x[:,1:3] = np.genfromtxt('../ass1_data/data/q3/logisticX.csv',delimiter=',')
x[:,1:3] = (x[:,1:3]-np.mean(x[:,1:3]))/np.std(x[:,1:3])
theta = np.zeros(3)
cost1 = cost_function(x, y, theta)
m, n = x.shape
while True:
    h = sigmoid(np.dot(x, theta))
    gradient = np.dot(x.T, (h - y)) / y.size
    diag = np.multiply(h, (1 - h)) * np.identity(m)
    hessian = (np.dot(np.dot(x.T, diag), x))/m
    theta = theta - np.dot(np.linalg.inv(hessian), gradient)
    cost2 = cost_function(x, y, theta)
    if cost2 >= cost1:
        break
    else:
        cost1 = cost2
print(theta)
x1 = []
x0 = []
for i in range(len(x)):
    if y[i] == 1.:
        x1.append(x[i])
    else:
        x0.append(x[i])
x1 = np.array(x1)
x0 = np.array(x0)
plt.scatter(x1[:,1],x1[:,2],marker= 'o',label = "y=1")
plt.scatter(x0[:,1],x0[:,2],marker= 'v',label = 'y=0')
plt.plot(x[:,1], (0.5-theta[0]-theta[1]*x[:,1])/theta[2], linestyle='solid',label = "decision boundary")
plt.legend(loc='upper left')
plt.show()
