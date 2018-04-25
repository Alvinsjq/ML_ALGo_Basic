#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 11:16:51 2018

@author: alvin
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io # 用来处理matlab中的mat数据文件类型
import scipy.misc # 用以将矩阵用图像展示
import matplotlib.cm as cm # Used to display images in a specific colormap
import random # 随机选取图片来展示
from scipy.special import expit # 向量化的sigmod函数


# =========== Part 1: 加载并可视化数据 =============


datapath = 'data/ex3data1.mat'
data = scipy.io.loadmat(datapath)
X = data['X']
y = data['y']
print(X.shape)
print(y.shape)

def _display_data():
    '''
    # 这里随机展示100条训练数据，并将它们以图片的形式展现
    - 每行数据需要重新排列为20 * 20 的矩阵
    - 100个数据分成10 * 10 排列
    '''
    
    all_fig = np.zeros((10*20,10*20)) # 整个需要展示的图片矩阵的大小是200 * 200规格的
    index_of_samples = random.sample(range(X.shape[0]),100) # 从5000个样本中随机跳出100条展示
    
    row, col = 0, 0
    for i in index_of_samples:
        if col == 10:
            row += 1
            col = 0 
        fig = X[i].reshape(20,20).T
        # 下面将每个小矩阵赋值到大矩阵的对应部分
        all_fig[row * 20:(row+1)*20,col * 20:(col+1)*20] = fig
        col += 1
        
    plt.figure(figsize=(8,8))
    img = scipy.misc.toimage(all_fig)
    plt.imshow(img, cmap = plt.cm.gray_r)
    
_display_data()

# ============ Part 2a: Vectorize Logistic Regression ============

def hpy_sigmod_fucntion(X_inter,theta_inter):
    '''
    # 先定义好假设函数 h = X * theta ==> (5000,401) * (401 * 1) = (5000,1)
    - X_inter 增加了一列1后的X
    - theta_inter 增加了一行 1后的theta
    '''
    return expit(np.dot(X_inter,theta_inter))
    
def LR_Costfunction(theta_inter,X_inter,y,lamada=0.):
    '''
    计算损失函数
    '''
    m = X_inter.shape[0] # 得到样本的数量
    hyp = hpy_sigmod_fucntion(X_inter,theta_inter) # 得到假设函数\    
    reg = np.dot(theta_inter.T,theta_inter) * (lamada / (2 * m))
    J = np.dot(y.T,np.log(hyp))+np.dot((1 - y.T),np.log(1 - hyp)) # (1,1)
    return J + reg


def Gradient(theta_inter,X_inter,y,lamada=0.):
    '''
    计算梯度下降的值
    '''
    m = X_inter.shape[0] # 得到样本的数量
    hyp = hpy_sigmod_fucntion(X_inter,theta_inter) # 得到假设函数
    hyp = np.asarray(hyp).reshape(hyp.shape[0],1)
    h_y = hyp - y # 5000 * 1
    reg = theta_inter[1:] * (lamada / m) 
    reg = np.asarray(reg).reshape(reg.shape[0],1)
    grad = (1 / m) * np.dot(X_inter.T,h_y) # 401 * 1 
    grad[1:] = grad[1:] + reg
    return grad # 401 * 1
    
def opt_Cost(theta,X,y,lamada=0.):
    '''
    利用优化算法，得到优化结果，也就是cost值和grad值
    '''
    from scipy import optimize
    res = optimize.fmin_bfgs(LR_Costfunction, theta, fprime=Gradient, args=(X,y,lamada) )
    return result[0], result[1]


def training_OnevsAll_theta(X,y,num_labels,lamada=0.):
    '''
    # One VS All 分类器是为每一个类别i预测样本为类别i的可能性，
    然后输入一个x，用训练好的模型做预测，选择概率最大的那个类别。
    - one vs all 多次训练logistics回归，并返回所有分类器的theta，
    得到theta矩阵all_theta，该矩阵的第i行代表对第i类的分类器
    '''
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels,n+1))
    
    X = np.hstack((np.ones((m,1)),X)) 
    
    for c in range(num_labels):
        '''
        为每一个类别c进行训练，得到theta_c
        '''
        print("Training theta for class %d" %c)
        initial_theta = np.zeros((n+1,1))
        theta,cost = opt_Cost(initial_theta,X,y,lamada)
        all_theta[c] = theta
        
    print("Finished!")
    
trained_theta = training_OnevsAll_theta(X,y,10,0.1)

   
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    