import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv

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
        theta[i] = theta[i] - eta*diff(x,y,theta,i)
    return (theta)

y = np.genfromtxt('../ass1_data/data/q1/linearY.csv',delimiter=',')
x = np.ones((len(y),2))
x[:,1] = np.genfromtxt('../ass1_data/data/q1/linearX.csv',delimiter=',')
x[:,1] = (x[:,1]-np.mean(x[:,1]))/np.std(x[:,1])
theta  = np.zeros(2)
costold = cost_func(x,y,theta)
costnew = 0
n = 0
eta = 0.1
theta0 = np.array([])
theta1 = np.array([])
cost = np.array([])
ms = np.linspace(-0.2, 2.0, 100)
bs = np.linspace(-1, 1, 100)
M, B = np.meshgrid(ms, bs)
zs = np.array([cost_func(x, y, np.array([mp, bp])) for mp, bp in zip(np.ravel(M), np.ravel(B))])
print(zs)
Z = zs.reshape(M.shape)
"""
            For contour plot
fig = plt.figure()
CS = plt.contour(M, B, Z, 25)
plt.xlabel('Theta0 -->')
plt.ylabel('Theta1 -->')
plt.clabel(CS, inline=1, fontsize=10)
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.4, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('theta(0)')
ax.set_ylabel('theta(1)')
ax.set_zlabel('cost')
while  (costold - costnew) > 0.000001:
    theta0 = np.append(theta0,np.array([theta[0]]))
    theta1 = np.append(theta1,np.array([theta[1]]))
    cost = np.append(cost,np.array([(cost_func(x,y,theta))]))
    print(theta0)
    """
    contour plot line
    plt.plot(theta0, theta1, linestyle='solid',color='r', marker='x', label='Optimal Value')
    """

    ax.plot3D(theta0, theta1,cost,'red')
    plt.pause(0.2)
    costold = cost_func(x,y,theta)
    theta = update_theta(x,y,theta,eta)
    costnew = cost_func(x,y,theta)
    n+=1
print(n)
print(eta)
print(theta)


"""
line graph plot
print(costnew)
plt.xlabel('x')
plt.ylabel('y')

plt.scatter(x[:,1],y,label = "datapoints")
plt.plot(x[:,1], np.dot(x,theta), linestyle='solid', label = "learned model",color = 'g')
plt.legend(loc='upper left')
"""
plt.savefig('Fig2.png')
plt.show()