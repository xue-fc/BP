import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)

def draw(a,l):
    pre_pos = a[l == 1, :2]
    pre_neg = a[l == 0, :2]
    plt.scatter(pre_pos[:, 0], pre_pos[:, 1], color='r', label='pos')
    plt.scatter(pre_neg[:, 0], pre_neg[:, 1], color='b', label='neg')
    # plt.title('Epoch:'+str(epoch)+' Acc:'+str(acc))
    plt.title("data point")
    plt.legend()
    plt.show()
    return

def sigmoid(x):
    for i in range(len(x)):
        x[i] = 1.0/(1+math.exp(-x[i]))
    return x

def dsigmoid(x):
    return x*(1-x)

def vector2dia(x):
    t = np.zeros([len(x),len(x)])
    for i in range(len(x)):
        t[i][i] = x[i]
    return t

def square(x):
    return np.dot(vector2dia(x),x)

class bp:
    def __init__(self,ni,nh,no):
        self.ni_ = ni + 1 #加的这个1就是bias
        self.nh_ = nh
        self.no_ = no

        self.ui = np.zeros(self.ni_ * self.nh_).reshape((self.ni_,self.nh_))
        self.vi = np.zeros(self.ni_ * self.nh_).reshape((self.ni_,self.nh_))
        self.uo = np.zeros(self.no_ * self.nh_).reshape((self.nh_,self.no_))
        self.vo = np.zeros(self.no_ * self.nh_).reshape((self.nh_,self.no_))


        #第一层的权重
        one1 = np.ones(self.ni_ * self.nh_).reshape((self.ni_,self.nh_))
        self.ui = np.random.rand(self.ni_ * self.nh_).reshape((self.ni_,self.nh_))
        self.ui = 2 * self.ui - one1
        self.vi = np.random.rand(self.ni_ * self.nh_).reshape((self.ni_,self.nh_))
        self.vi = 2 * self.vi - one1
        #第二层的权重
        one2 = np.ones(self.no_ * self.nh_).reshape((self.nh_,self.no_))
        self.uo = np.random.rand(self.no_ * self.nh_).reshape((self.nh_,self.no_))
        self.uo = 2 * self.uo - one2
        self.vo = np.random.rand(self.no_ * self.nh_).reshape((self.nh_,self.no_))
        self.vo = 2 * self.vo - one2
        #中间变量
        self.ai = np.zeros(self.ni_).reshape(self.ni_,1)
        self.ah = np.zeros(self.nh_).reshape(self.nh_,1)
        self.ao = np.zeros(self.no_).reshape(self.no_,1)



    def online_train(self,eta=0.1,iter=1000):
        self.eta = eta
        X = []
        y = []
        fr = open('train.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            X.append([float(linearr[0]), float(linearr[1]), 1])
            y.append([int(linearr[2])])
        n = np.shape(X)[0]    #数据集长度

        errors = []
        for i in range(iter):   #每次循环对数据集里每条数据进行训练
            error = 0.0
            for j in range(n):
                self.forward(X[j])
                error = error + self.backward(y[j])
            errors.append(error[0])
            if i % 100 == 0:
                 print("After", i, "iterations,the total error of the training set is", error)
        # plt.plot(errors)
        # plt.title("eta = 0.001")
        # plt.xlabel('epoch')
        # plt.ylabel('error')
        # plt.show()

    def forward(self,X):
        self.ai = np.array(X).reshape(len(X),1)
        self.ah = sigmoid(np.dot(self.ui.T,square(self.ai)) + np.dot(self.vi.T,self.ai))
        self.ao = sigmoid(np.dot(self.uo.T,square(self.ah)) + np.dot(self.vo.T,self.ah))
        #self.ao = sigmoid(np.dot(self.ah*self.ah,self.uo) + np.dot(self.ah,self.vo))
        return self.ao

    def backward(self,y):
        error = y - self.ao

        s2 = 2 * error * dsigmoid(self.ao)
        s1 = s2 * np.dot(vector2dia(dsigmoid(self.ah)), (2 * np.dot(vector2dia(self.ah), self.uo) + self.vo))
        #s0 = np.dot(vector2dia(dsigmoid(self.ai)), (2 * np.dot(vector2dia(self.ai),self.ui) + self.vi)).dot(s1)

        self.uo = self.uo + self.eta * square(self.ah).dot(s2.T)
        self.vo = self.vo + self.eta * self.ah.dot(s2.T)

        self.ui = self.ui + self.eta * square(self.ai).dot(s1.T)
        self.vi = self.vi + self.eta * self.ai.dot(s1.T)


        error = 0.5*error**2
        return error

    def test(self):
        atributemat = []; labelmat = []
        fr = open('test.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            atributemat.append([float(linearr[0]),float(linearr[1]),1])
            labelmat.append(int(linearr[2]))
        a = np.array(atributemat)
        l = np.array(labelmat)
        draw(a,l)
        n = np.shape(atributemat)[0]
        errcount = 0
        res = []
        for i in range(n):
            temp = self.forward(atributemat[i])[0]
            if((labelmat[i] == 1 and temp < 0.5) or (labelmat[i] == 0 and temp > 0.5)):
                errcount = errcount + 1
        print('the total error rate of the test set is %f' % (errcount/(n*1.0)))
        return (errcount/(n*1.0))

def main():
    n1 = bp(2,10,1)
    n1.online_train(0.0001,1000)
    n1.test()
    return

main()