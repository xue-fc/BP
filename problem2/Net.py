import numpy as np
import matplotlib.pyplot as plt

class Adaline(object):
    # 初始化
    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    # 训练模型
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        self.cost_ = []
        self.accList_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)

            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum()/2.0
            self.cost_.append(cost)
            p_out = self.predict(X)
            acc = np.sum(y == p_out) / 700
            self.accList_.append(acc)
        plt.plot(self.accList_)
        plt.xlabel('epoch')
        plt.ylabel('Acc')
        plt.title('eta = 0.1')
        plt.show()
        print(self.accList_)
        return self


    # 输入和权值的点积,即公式的z函数,图中的net_input
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    # 线性激活函数
    def activation(self,X):
        return self.net_input(X)

    # 利用阶跃函数返回分类标签
    def predict(self,X):
        return np.where(self.activation(X)>=0.0,1,-1)

    def plot_errors(self):
        print(self.cost_)
        plt.plot(np.log10(self.cost_))
        plt.xlabel('epoch')
        plt.ylabel('log(errors)')
        plt.title('eta = 0.1')
        plt.show()
