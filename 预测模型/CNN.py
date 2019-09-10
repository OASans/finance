"""
    CNN模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



"""
    下面的函数将原始表示转化为训练集，用于CNN模型
    
    原始表示(类型：python list)：[[a(1), a(2), ... , a(T)],
                                   ...
                                  [z(1), z(2), ... , z(T)]]
            其中每个list表示某个因子连续T天的数据
            
    训练集(类型：python list，一共T-N个样本)：[[a(1), a(2), ... , a(N)],   [ a(N+1),
                                                ...                          ...
                                               [z(1), z(2), ... , z(N)]]     z(N+1)]
"""
def CNN_rawset2trainset(rawset, N=30):
    """
        rawset: 原始表示
        N：训练集中样本数据的时间长度
        
        返回：trainset ndarray，其中每个元素都是一个样本，一共T-N个
              trainlabel ndarray，其中每个元素都是一个样本，一共T-N个 
    """
    T = len(rawset[0])
    rawset = np.array(rawset)
    
    trainset = []
    trainlabel = []
    
    for i in range(T-N):
        trainset.append(rawset[:, i:i+N])
        trainlabel.append(rawset[:, i+N])
        
    return np.array(trainset), np.array(trainlabel)
    
 
"""
    CNN模型的定义
    该CNN模型包含两个卷积层以及三个全连接层，可以由前几天的数据预测未来的收益率
    
    目前该网络的输入为[T-N, 1, 5, 30] T-N为样本数，1为通道个数，30和5分别为图片的长和宽
    如果想要输入不同长宽的图片，想要修改网络参数
    网络的输出为[T-N, 1]，
""" 
class CNN(nn.Module):
    def __init__(self, num_of_factors=5):   # num_of_factors: 因子个数
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3456, 1028),
            nn.PReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1028, 512),
            nn.PReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, num_of_factors),         # 直接预测收益率
            nn.PReLU(),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
        
        
"""
    CNN模型的训练函数
"""
def CNN_train(model, xtrain, ytrain):
    criterion = nn.MSELoss(reduction='sum')

    # 优化器
    optimizer = optim.LBFGS(cnn.parameters(), lr=0.1)

    # 开始训练    
    for i in range(5):
        print('STEP: ', i)
    
        def closure():
            optimizer.zero_grad()
            out = cnn(xtrain)
            loss = criterion(out, ytrain)
            print('loss: %5.3f  ' %(loss.item()))
            loss.backward()
        
            return loss
    
        optimizer.step(closure)
       
    return model
    
    
"""
    利用CNN模型进行预测
    
    model: 训练好的CNN模型
    sample: 类型为python list，形状为 [[a(1), ... , a(i)],
                                        ...
                                       [z(1), ... , z(i)]]    i=30
                          返回值预测  [a(i+1),
                                        ...
                                       z(i+1)]
"""
def CNN_predict(model, sample):
    # 对输入进行预处理
    sample = torch.from_numpy(np.array(sample))
    sample = sample.unsqueeze(0).unsqueeze(0)   # 变成单通道图片，并且batch大小为1
    
    with torch.no_grad():
        pred = cnn(sample)
    
    return pred[0].tolist()                # 返回预测值（即最后一组输出）    
        
        
        
if __name__ == "__main__":
    rawset = list(np.random.rand(5,40))
    trainset, trainlabel = CNN_rawset2trainset(rawset, 30)

    # 将ndarray转为tensor
    trainset = torch.from_numpy(trainset)
    trainlabel = torch.from_numpy(trainlabel)
    
    # 将训练数据变成单通道图片
    trainset = trainset.unsqueeze(1)
    
    """
    print(trainset.shape)
    print(trainlabel.shape)
    """
    
    # 建立CNN模型
    cnn = CNN().double()
    
    # 测试模型有无错误
    out = cnn.forward(trainset)
    print(out.shape)
    
    # 训练CNN模型
    cnn = CNN_train(cnn, trainset, trainlabel)
    
    # 利用CNN模型进行预测
    sample = list(np.random.rand(5,30))
    label = CNN_predict(cnn, sample)
    print(label)
    