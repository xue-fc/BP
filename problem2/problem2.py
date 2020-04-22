import torch
import numpy as np
import torch.nn as nn
from Net import Adaline
import torch.optim as optim
import matplotlib.pyplot as plt
from Mydataset import myDataSet
#4.10划分数据集还遇到了一些问题，不能把label和data放一起，但怎么分开呢？
#4.14划分问题解决，自己重写一个dataset即可设定data和label
#4.14上面走弯路了，这个问题里不需要用torch的dataset，直接用矩阵输入即可，用shuffle打乱，然后输入
#4.14adaline中必须要加个阶跃函数，不然不会收敛，不知道为什么网上并没有这部分内容
#4.14发现问题了，不能加阶跃函数，可能是之前learning rate没设置好，导致不收敛
#正负数据集
x_pos = np.random.uniform(low=0, high=1, size=500)
noise = np.random.uniform(low=0.1, high=0.2, size=500)
y_pos = -1 * x_pos + 1 - noise
label_pos = 0 - np.ones(500)

x_neg = np.random.uniform(low=0, high=1, size=500)
noise = np.random.uniform(low=0.1, high=0.2, size=500)
y_neg = -1 * x_neg + 1 + noise
label_neg = np.ones(500)

x_train = np.hstack([x_pos, x_neg])
y_train = np.hstack([y_pos, y_neg])

full_label = np.hstack([label_pos, label_neg])
full_set = np.vstack([x_train, y_train, full_label])
full_set = full_set.T
np.random.shuffle(full_set)

train_set = full_set[:700,:2]
train_label = full_set[:700,2]

test_set = full_set[700:,:2]
test_label = full_set[700:,2]

def plotAcc(acc_list):
    plt.plot(acc_list)
    plt.title("Training process")
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()

def drawBoundary(sample, pre_label):
    pre_pos = sample[pre_label==-1]
    pre_neg = sample[pre_label==1]
    plt.scatter(pre_pos[:,0], pre_pos[:,1], color='r', label='pos')
    plt.scatter(pre_neg[:,0], pre_neg[:,1], color='b', label='neg')
    plt.title("eta = 0.01")
    plt.legend()
    plt.show()

def main():
    drawBoundary(full_set[:,:2],full_set[:,2])
    net = Adaline(0.1, 100)
    net.fit(train_set, train_label)
    #print(net.predict(full_set))
    net.plot_errors()

    #drawBoundary(test_set,net.predict(test_set))

main()
