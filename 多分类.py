
# coding: utf-8

# # 多分类

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
from numpy.linalg import cholesky
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 生成样本数据
# 

# In[31]:


def genData(M, N = 100):
    rnd.seed(0)
    Sigma = np.array([[1, 0], [0, 1]])
    R = cholesky(Sigma)
    
    mu = np.array([[M, 0]])
    s = np.dot(np.random.randn(N, 2), R) + mu
    
    one = np.ones(s.shape[0]).reshape(100,1)
    s = np.concatenate((s,  M * one), axis = 1)
    
    return s


# ## 绘图函数

# In[26]:


def genPlt(S, w, style, plt):
    y = np.linspace(-3,3)
    x = (w[0] + w[2] * y)/w[1]

    plt.plot(S[:,0],S[:,1],style)
#     plt.plot(S[101:,0],S[101:,1],'o')
    plt.plot(x,y)
    
    return plt


# ## 感知器函数

# In[4]:


def PLA(T, wi):
    
    study_total = 100000 # 总训练次数
    study_step  = 0.001  # 训练步长
    w_total = 0          # w改变次数
    
    if(wi == 0):
        w = np.zeros(T.shape[1])  # 初始化权重向量为0 [权重都从0开始]
    else:
        w = np.ones(T.shape[1])   # 初始化权重向量为1 [权重都从1开始]
#     print(' W     X      W       B')
    #训练study_total次
    for study in range(study_total):
        w_before = w    #训练前的w值
        #训练
        for t in range(T.shape[0]):
            # 计算实际的y值，其期望值为T[0][2]
            X = T[t][0:T.shape[1]-1]   #X的值
            Y = T[t][T.shape[1]-1]     #期望值
            distin = Y * (w[0] + np.dot(w[1:],X))
            #print('sign:', np.sign((w[0] + np.dot(w[1:],X))), 'Y:', Y)
            #判断X是否是误分类点
            if distin <= 0:
                # 根据误差优化w的值
                w[1:] = w[1:] + study_step*Y*X
                w[0]  = w[0] + study_step*Y
#                 print('w',w_total,': x',t,w[0:w.shape[0]])
                w_total = w_total + 1
                flag = 0

        #经过训练后w、b都不在变化，说明训练集中已没有误分类点，那么跳出循环
        if w_before is w :
            print('训练后，得到w:', w[0:w.shape[0]])
            break        
    return w


# In[46]:


s = np.zeros((3,100,3))
w = np.zeros((3,3,3))
s[0,:,:] = genData(1)
s[1,:,:] = genData(5)
s[2,:,:] = genData(10)

for i in range(3):
    for j in range(3):
        s1 = s[i,:,:]
        s1[:,2] = 1
        s2 = s[j,:,:]
        s2[:,2] = -1
        train_data = np.concatenate((s1,s2), axis=0)
        w[i,j] = PLA(train_data, 0)
# w = PLA(s, 0)
print(w)
plt = genPlt(s[0,:,:], w[1,2], 'bs', plt)
plt = genPlt(s[1,:,:], w[0,1], 'r*', plt)
plt = genPlt(s[2,:,:], w[0,2], 'y^', plt)

