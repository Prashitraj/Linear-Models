import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
import math

data = []
with open('../ass1_data/data/q4/q4x.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split("  ")
        data.append([float(i) for i in k]) 

datax = np.array(data)
datax[:,0] = (datax[:,0]-np.mean(datax[:,0]))/np.std(datax[:,0])
datax[:,1] = (datax[:,1]-np.mean(datax[:,1]))/np.std(datax[:,1])
# 0 represents alaska and 1 represents canada 

data = []
with open('../ass1_data/data/q4/q4y.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip()
        data.append(float(0) if k == "Alaska" else float(1)) 

datay = np.array(data)
mu0 = np.mean(datax[datay[:,]==0], axis=0)
mu1 = np.mean(datax[datay[:,]==1], axis=0)
phi = np.mean(datay)
x_0 = datax[datay[:,]==0]
x_1 = datax[datay[:,]==1]
x0 = datax[datay[:,]==0] - mu0
x1 = datax[datay[:,]==1] - mu1
# linear boundary seperation
sigma = ((x0.T).dot(x0)+(x1.T).dot(x1))/datax.shape[0]
print(sigma)
print(mu0)
print(mu1)
sigma = np.array(sigma,dtype = 'float')
sigmainv = np.linalg.inv(sigma)
k = 2*(sigmainv).dot(mu1-mu0)
b = (mu1.T).dot(sigmainv).dot(mu1) - (mu0.T).dot(sigmainv).dot(mu0) + np.log(phi) - np.log(1-phi)
plt.scatter(x_1[:,0],x_1[:,1],marker= 'o',label = "Canada")
plt.scatter(x_0[:,0],x_0[:,1],marker= 'v',label = "Alaska")
plt.plot(datax[:,0],(-b-(k[0]/k[1])*datax[:,0]) , linestyle='solid',label = "linear decision boundary", c = 'r')
sigma0 = (x0.T).dot(x0)/(datax[datay[:,]==0]).shape[0]
sigma1= (x1.T).dot(x1)/(datax[datay[:,]==1]).shape[0]
print(sigma0)
print(sigma1)

C = np.log(np.linalg.det(sigma0)) - np.log(np.linalg.det(sigma1)) - 2*(np.log(phi)-np.log(1-phi))
[[a,b],[c,d]] = np.linalg.inv(sigma1)
[[p,q],[r,s]] = np.linalg.inv(sigma0)
x0plot = np.linspace(-2.5,2.5,500)
x1plot = []
x1plotn = []
for x in x0plot:
    u=d-s
    v = -2.*d*mu1[1]+2.*s*mu0[1] + b*x-b*mu1[0] + c*x-c*mu1[0] - q*x+q*mu0[0] - r*x+r*mu0[0]
    w=C-a*((x-mu1[0])**2)+p*((x-mu0[0])**2)+b*mu1[1]*x+c*mu1[1]*x-q*mu0[1]*x-r*mu0[1]*x+d*(mu1[1]**2)-s*(mu0[1]**2)-b*mu1[0]*mu1[1]-c*mu1[0]*mu1[1]+q*mu0[1]*mu0[0]+r*mu0[1]*mu0[0]
    y = (-v+math.sqrt(v**2+4*u*w))/(2*u)
    yn = (-v-math.sqrt(v**2+4*u*w))/(2*u)
    x1plot.append(y)
    x1plotn.append(yn)
plt.plot(x0plot,x1plot,label = "quadratic decision boundary", c= 'b')
plt.plot(x0plot,x1plotn,label = "quadratic decision boundary", c= 'b')
plt.legend(loc='upper left')
plt.show()