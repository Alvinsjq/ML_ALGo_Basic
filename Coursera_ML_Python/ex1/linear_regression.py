#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:51:18 2018

@author: alvin
"""

import numpy as np
import matplotlib.pyplot as plt



# ==========函数部分=====================================

def _plot(x,y):
    plt.figure(figsize=(10,8));
    plt.plot(x,y,'rx',markersize=10);
    plt.grid(True);
    plt.xlabel('Population of City in 10,000s');
    plt.ylabel('Profit in $10,000s'); 
    
    
def _plot_J(J_histroy):
    plt.figure(figsize=(10,6));
    plt.plot(range(len(J_histroy)),J_histroy,'ro');
    plt.grid(True);
    plt.title("Convergence of Cost Function");
    plt.xlabel("Iterations");
    plt.ylabel("Cost:J");
    

    
def _ComputerCost(X,y,theta):
     #这里实现损失函数的计算方法，
     #theta是n维向量 以及 截距项 因此为 （n+1）*1
     #X是增加了一个截矩项的特征向量矩阵 m*(n+1)
     #y是对应真实的估计值向量 m*1
     m = y.size;
     hyp = np.dot(X,theta); # 这里的hyp是 (theta_0*x1 + theta_1)
     a = hyp - y;  
     coef = 1./(2*m);
     J = coef * np.dot(a.T,a);#返回一个实数
     return float(J); 

def _gradientDescent(X,y,theta,alpha,iterations):
     # 通过梯度下降来得到theta的估计值
     # 根据循环次数和学习率alpha来更新theta参数
      m = y.size;
      J_histroy = [];#初始化cost的数组，用于存放cost的值
      theta_histroy = [];
      #梯度下降公式实现
     
      for iter in range(iterations) :
          a = np.dot(np.transpose(X),np.dot(X,theta)-y);
          theta_histroy.append(list(theta[:,0]))
          theta = theta - alpha * (1/m) * a;
          J_histroy.append(_ComputerCost(X,y,theta));#将每一次梯度下降的损失函数结果存起来
      return theta,J_histroy,theta_histroy;
        
    


## Main部分

datapath = "data/ex1data1.txt";
data = np.loadtxt(datapath,delimiter=',',usecols=(0,1),unpack=True);
X = np.transpose(np.array(data[:-1]));
y = np.transpose(np.array(data[-1:]));
m = y.size;


# =====================可视化数据点=======================

_plot(X,y);

# =====================损失函数==================

b = np.ones((X.shape[0],1));
X = np.concatenate((X,b),axis=1); # X此时为 m*(n+1),注意，这边是先X再1，这会影响到theta的系数的分配
init_theta = np.zeros((X.shape[1],1)) #  初始化的theta为 (n+1)*1

J = _ComputerCost(X,y,init_theta);
print(J);
    
   
# =======================梯度下降==================
    
iterations = 1500;#参数设置
alpha = 0.01; 
theta,J_histroy, theta_histroy= _gradientDescent(X,y,init_theta,alpha,iterations);
print(theta);    

_plot_J(J_histroy);
_plot(X[:,0],y);
plt.plot(X[:,0],theta[0]*X[:,0]+theta[1],'b-'); # 画出hyp函数(theta_0*x1 + theta_1)


# =======================可视化J函数==================
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

    
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)
    
J_val = np.zeros((len(theta0_vals),len(theta1_vals)));
np.reshape(J_val,(len(theta0_vals),len(theta1_vals)));

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta1_vals[i]],[theta0_vals[j]]])
        J_val[i][j] = _ComputerCost(X,y,t)
        
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax.scatter(theta0_vals,theta1_vals,J_val,c=np.abs(theta0_vals),cmap=plt.get_cmap('YlGnBu'))
   
# =======================可视化梯度下降过程==================
plt.plot([x[1] for x in theta_histroy],[x[0] for x in theta_histroy],J_histroy,'bo-')    
plt.show()
    
    
    