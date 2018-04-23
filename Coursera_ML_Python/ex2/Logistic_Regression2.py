#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:05:38 2018

@author: alvin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
# 正规化的logistic回归



def _plot_data2(X,y):
    pos = X[np.where(y==1)[0]]
    neg = X[np.where(y==0)[0]]
    plt.figure(figsize=(10,8))
    plt.grid(True)
    plt.plot(pos[:,0],pos[:,1],'k+',label='y = 1')
    plt.plot(neg[:,0],neg[:,1],'go',label='y = 0')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    
def _plot_decision_boundary(theta,X_map,y,lamda,maxiter):
    
    '''
    result = optimize.fmin(_CostFunction_J, x0=initial_theta, args=(X_map, y, 10), maxiter=400, full_output=True)
    原来的方法会超出最大的迭代次数，因此会影响到收敛，因此使用optimize.minimize
    '''
    
    result = optimize.minimize(_CostFunction_J, theta, args=(X_map, y, lamda),  method='BFGS', options={"maxiter":maxiter, "disp":False}) 
    theta = result.x  
    print("以下是theta")
    print(theta)
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    # 在这个网格上，求出z
    for i in range(len(u)):
        for j in range(len(v)):
            map_ij = _mapFeature(np.array([u[i]]),np.array([v[j]]))
            z[i][j] = np.dot(map_ij,theta)
    z = np.transpose(z)
    
    _plot_data2(X,y)
    plt.contour(u,v,z,[0])
    plt.title("Decision Boundary with lamda = %d" %lamda)
    
    
def _sigmod(Z):
    g = np.zeros(Z.size)
    a = np.exp(-1 * Z) + 1
    g = 1 / a
    return g

def _CostFunction(theta,X,y,_lambda=0.):
    # 这里的logitisc回归的损失函数与之前的有所不同【推导】
    # 因此在ng的讲义中提出了概率假设的前提，因此这里的假设函数就是最大似然函数
    J = 0
    grad = np.zeros(theta.size)
    m = X.shape[0]
    h_theta = _sigmod(np.dot(X,theta)) # X*theta 为一个 m * 1 的向量
    # 由于h_theta是 m*1 自然y需要转置才能取得一个实数值
    cost = -1 * np.dot(np.transpose(y),np.log(h_theta)) - np.dot(np.transpose(1-y),np.log(1-h_theta))
    
    # 标准化部分 【公式的由来结合coursera上的ppt】
    theta_exp_0 = theta[1:].reshape((theta[1:].shape[0],1)) #  注意它现在是 n-1 * 1
    
    reg_cost = (_lambda/(2 * m)) * np.dot(theta_exp_0.T,theta_exp_0)
    
    reg_grad = theta_exp_0
    reg_grad_zero = np.zeros((1,reg_grad.shape[1]))
    reg_grad = (_lambda/ m) * np.concatenate((reg_grad_zero,reg_grad),axis=0)    
    J = (1/m) * cost + reg_cost
    grad = (1/m) * np.dot(np.transpose(X),(h_theta - y)) + reg_grad
    print (J)
    return J,grad

def _CostFunction_J(theta,X,y,_lambda=0.):
    J = 0 
    h_theta = _sigmod(np.dot(X,theta)) # X*theta 为一个 m * 1 的向量 
    t1 = np.dot(-np.array(y).T,np.log(h_theta))
    t2 = np.dot((1-np.array(y)).T,np.log(1-h_theta))
    reg = (_lambda/2) * np.sum(np.dot(theta[1:].T,theta[1:]))
    cost = -1 * np.dot(np.transpose(y),np.log(h_theta)) - np.dot(np.transpose(1-y),np.log(1-h_theta))
    J = float( (1./m) * ( np.sum(t1 - t2) + reg ) )
    return J
    
def _mapFeature(X1,X2):
    # 输入两个特征向量
    # 返回最高次为六的特征向量
    degrees = 6
    out = np.ones((X1.shape[0],1))
    for i in range(1,degrees+1):
        for j in range(i+1):
            t1 = X1 ** (i-j)
            t2 = X2 ** (j)
            b = (t1 * t2).reshape(t1.shape[0],1)
            out = np.concatenate((out,b),1);
            
    return out

# 画出数据
datapath = "ex2data2.txt"
data = np.loadtxt(datapath,delimiter=',',usecols=(0,1,2),unpack=True)
X = np.transpose(np.array(data[:-1]))
y = np.transpose(np.array(data[-1:]))
m = y.size
_plot_data2(X,y)
plt.show()

# 特征转换

  # 之前的logistics回归只能得到线性边界，然后在现在的数据集中是不存在合适的线性边界的，为了能够使得拟合效果更好
  # 我们需要使用较多的特征，因此这里需要进行特征转化，个数取决于多项式特征数量
X_map = _mapFeature(X[:,0],X[:,1])
initial_theta = np.zeros((X_map.shape[1],1))

J, grad = _CostFunction(initial_theta,X_map,y,1)
print(J)
print(grad[0:5])


test_theta = np.ones((X_map.shape[1],1))
J, grad = _CostFunction(test_theta,X_map,y,10)
print(J)
print(grad[0:5])

# 使用fmin来学习参数
result = optimize.minimize(_CostFunction_J,initial_theta,args=(X_map,y,0),method='BFGS',options={'maxiter': 400, 'disp': False})
op_theta = result.x
op_cost = result.fun


# 得到不同lamada的图像
_plot_decision_boundary(op_theta,X_map,y,0,500)
_plot_decision_boundary(op_theta,X_map,y,0,400)
_plot_decision_boundary(op_theta,X_map,y,1,400)
_plot_decision_boundary(op_theta,X_map,y,10,400)
_plot_decision_boundary(op_theta,X_map,y,100,400)






















