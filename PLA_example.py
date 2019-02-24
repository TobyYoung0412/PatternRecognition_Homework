#coding:utf-8
import numpy as np
 
class Perceptron(object):
    def __init__(self):
        self.study_step = 1    #学习步长即学习率
        self.study_total = 11   #学习次数即训练次数
        self.w_total = 1  #w更新次数
    #对数据集进行训练
    def train(self, T):
        w = np.zeros(T.shape[1]-1)  # 初始化权重向量为0 [权重都从0开始]
        b = 0  # 初始化阈值为0
        print(' W     X      W       B')
        #训练study_total次
        for study in range(self.study_total):
            w_before = w    #训练前的w值
            b_before = b    #训练前的b值
            #训练
            for t in range(T.shape[0]):
                # 计算实际的y值，其期望值为T[0][2]
                X = T[t][0:T.shape[1]-1]   #X的值
                Y = T[t][T.shape[1]-1]     #期望值
                distin = Y*self.input_X(X, w, b)
                #判断X是否是误分类点
                if distin <= 0:
                    w = w + self.study_step*Y*X
                    b = b + self.study_step*Y
                    print('w',self.w_total,': x',t+1,w[0:w.shape[0]], '  ', b)
                    self.w_total = self.w_total + 1

            #经过训练后w、b都不在变化，说明训练集中已没有误分类点，那么跳出循环
            if w_before is w and b_before == b:
                print('训练后，得到w、b:', w[0:w.shape[0]], ' ', b)
                break
        return w,b
    #得出w*x+b的值
    def input_X(self, X, w, b):
        return np.dot(X,w) + b  #wwww**
 
    #由X去预测Y值
    def prediction(self, X, w, b):
        Y = self.input_X(X, w, b)
        return np.where(Y >= 0, 1, -1)
 
 
if __name__ == '__main__':
    per = Perceptron()
    #训练数据集，x1=(3,3),x2=(4,3),x3=(1,1), 对应于y1=1,y2=1,y3=-1
    T = np.array([[3,3,1],[4,3,1],[1,1,-1]])  #进行训练的数据集
    # w,b = per.train(T)     #经过训练得到w\b
 
    X = np.array([3, 3])  # 对X进行预测
    Y = per.prediction(X,w,b)   #得到X的预测值
    print('X预测得到Y：',Y)