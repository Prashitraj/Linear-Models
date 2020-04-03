import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
import math

def cost_func(x,y,theta):
    n = len(y)
    error = y-np.dot(x,theta)
    cost = np.sum(error**2)
    cost = cost/(2*n)
    return cost

def diff(x,y,theta,k):
    n = len(y) 
    error = np.dot(x,theta)-y
    error = np.dot(error,x[:,k])
    sum = np.sum(error)
    return (sum/(2*n))

def update_theta(x,y,theta,eta):
    for i in range(len(theta)):
        theta[i] = theta[i]-eta*diff(x,y,theta,i)
    return theta


def updatenew(xtrain,ytrain,thetanew,r):
    costold = cost_func(xtrain,ytrain,thetanew)
    costnew = 0
    n = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('theta(0)')
    ax.set_ylabel('theta(1)')
    ax.set_zlabel('theta(2)')
    theta0 = np.array([])
    theta1 = np.array([])
    theta2 = np.array([])
    while abs(costold-costnew)>0.000005:    
        i = 0
        n+=1
        costold = cost_func(xtrain,ytrain,thetanew)
        theta0 = np.append(theta0,np.array([thetanew[0]]))
        theta1 = np.append(theta1,np.array([thetanew[1]]))
        theta2 = np.append(theta2,np.array([thetanew[2]]))
        while i < 1000000:
            x = xtrain[i:i+r]
            y = ytrain[i:i+r]
            eta = 0.05
            thetanew = update_theta(x,y,thetanew,eta)
            i+=r
        costnew = cost_func(xtrain,ytrain,thetanew)
    # theta0 = np.append(theta0,np.array([thetanew[0]]))
    # theta1 = np.append(theta1,np.array([thetanew[1]]))
    # theta2 = np.append(theta2,np.array([thetanew[2]]))
    print(theta0)
    print(theta1)
    print(theta2)
    ax.plot3D(theta0, theta1,theta2,'red')
    plt.show()
    return thetanew,n


# generating 1M datapoints according to hypothesis
xtrain = np.ones((1000000,3))
xtrain[:,1] = np.random.normal(3,2,1000000)
xtrain[:,2] = np.random.normal(3,2,1000000)
theta = np.array((3,1,2))
ytrain = np.ones(1000000)
e = np.random.normal(0,math.sqrt(2),1000000)
ytrain = np.dot(xtrain,theta)+e
q2train = np.ones((1000000,4))
q2train[:,0:3] = xtrain
q2train[:,3] = ytrain
# shuffling datapoints
np.random.shuffle(q2train)
xtrain = q2train[:,0:3]
ytrain = q2train[:,3]
thetanew = np.zeros(3)
thetanew,n = updatenew(xtrain,ytrain,thetanew,1000000)
print (thetanew)
print(n-1)
q2test = np.genfromtxt('../ass1_data/data/q2/q2test.csv',delimiter=',')
xtest = np.ones((len(q2test)-1,3))
xtest[:,1:3] = q2test[1:,0:2]
ytest = q2test[1:,2]
cost = cost_func(xtest,ytest,thetanew)
# cost learned
print(cost)
# cost hypo
cost = cost_func(xtest,ytest,theta)
print(cost)