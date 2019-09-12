"""
    LSTM模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


"""
    下面的函数将原始表示转化为训练集，用于LSTM模型
    
    原始表示(类型：python list)：[[a(1), a(2), ... , a(T)],
                                   ...
                                  [z(1), z(2), ... , z(T)]]
            其中每个list表示某个因子连续T天的数据
            
    训练集(类型：numpy adarray，一共T-N个样本)[[a(1), a(2), ... , a(N)], [[a(2), ... , a(N+1)],
                                                ...                        ...
                                               [z(1), z(2), ... , z(N)]]  [z(2), ... , z(N+1)]]
"""
def LSTM_rawset2trainset(rawset, N):
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
        trainlabel.append(rawset[:, i+1:i+N+1])
        
    return np.array(trainset), np.array(trainlabel)
    
    
     
"""
    LSTM模型的定义
    
    模型参数：num_of_factors: 因子个数
              hidden_dim:     隐层神经元个数，越大则模型复杂度越高
              num_layers:     LSTM中lstm cell个数，越大则模型复杂度越高
              
    模型输入：[T-N, SEQ_LENGTH, num_of_factors] (N为样本个数)
    模型输出：[T-N, SEQ_LENGTH, num_of_factors]
    (输入输出的含义可见train_test_split函数)
    
    这里使用LSTM模型预测未来一天的多因子值
"""
class LSTM(nn.Module):
    def __init__(self, num_of_factors=3, hidden_dim=15, batch_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = num_of_factors
        self.output_dim = num_of_factors
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs):
        outputs = []
        
        for seq in inputs:
            lstm_out, self.hidden = self.lstm(seq.view(len(seq), self.batch_size, -1))
            y_pred = self.linear(lstm_out)
            outputs.append(y_pred)
        
        return torch.stack(outputs).squeeze(2).squeeze(2)  # 去掉冗余的维度
        
        
"""
    LSTM模型的训练函数
    
    model为定义好的LSTM模型
    xtrian与ytrain分别为训练数据与其标记，其形状要求与LSTM模型定义的输入输出形状相同
"""
def LSTM_train(model, xtrain, ytrain):
    # 损失函数
    criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.LBFGS(model.parameters(), lr=0.1)
    
    # 开始训练    
    for i in range(20):
        print('STEP: ', i)
    
        def closure():
            optimizer.zero_grad()
            out = model(xtrain)
            loss = criterion(out, ytrain)
            print('loss: %5.3f  ' %(loss.item()))
            loss.backward()
        
            return loss
    
        optimizer.step(closure)
        
    return model
    
    
"""
    利用LSTM模型进行预测
    
    model: 训练好的LSTM模型
    sample: 类型为python list，形状为 [[a(1), ... , a(i)],
                                        ...
                                       [z(1), ... , z(i)]]
                          返回值预测  [a(i+1),
                                        ...
                                       z(i+1)]
"""
def LSTM_predict(model, sample):
    # 先进行预处理
    sample = torch.from_numpy(np.array(sample))
    sample = sample.transpose(0, 1)             # seq_length在前，num_of_factors在后
    sample = sample.unsqueeze(0)                
    
    with torch.no_grad():
        pred  = model(sample)
    
    return pred[0].tolist()[-1]                 # 返回预测值（即最后一组输出）
 

def final_lstm_predict(rawset):
    trainset, trainlabel = LSTM_rawset2trainset(rawset, 2)

    # 将ndarray转为tensor
    trainset = torch.from_numpy(trainset)
    trainlabel = torch.from_numpy(trainlabel)

    # 将tensor的后两个维度进行交换，由[N, dim, SEQ_LENGTH]形式变成[N, SEQ_LENGTH, dim]形式
    trainset = trainset.transpose(1, 2)
    trainlabel = trainlabel.transpose(1, 2)

    # 构建LSTM模型
    lstm = LSTM(num_of_factors=5).double()

    # 测试模型有无错误
    out = lstm.forward(trainset)
    print(out.shape)

    # 训练LSTM模型
    lstm = LSTM_train(lstm, trainset, trainlabel)

    # 利用LSTM模型进行预测
    label = LSTM_predict(lstm, rawset)
    return label
        
if __name__ == "__main__":
    rawset = [[4.0,2.0,5.0,6.0,4.0,4.0,2.0,5.0,6.0,4.0,4.0,2.0,5.0,6.0,4.0,4.0,2.0,5.0,6.0,4.0,4.0,2.0,5.0,6.0,4.0],
              [8.0,9.0,7.0,6.0,9.0,8.0,9.0,7.0,6.0,9.0,8.0,9.0,7.0,6.0,9.0,8.0,9.0,7.0,6.0,9.0,8.0,9.0,7.0,6.0,9.0],
              [5.0,6.0,4.0,7.0,5.0,5.0,6.0,4.0,7.0,5.0,5.0,6.0,4.0,7.0,5.0,5.0,6.0,4.0,7.0,5.0,5.0,6.0,4.0,7.0,5.0],
              [4.2,5.4,6.5,7.8,9.8,4.2,5.4,6.5,7.8,9.8,4.2,5.4,6.5,7.8,9.8,4.2,5.4,6.5,7.8,9.8,4.2,5.4,6.5,7.8,9.8],
              [1.2,5.4,3.4,2.3,4.2,1.2,5.4,3.4,2.3,4.2,1.2,5.4,3.4,2.3,4.2,1.2,5.4,3.4,2.3,4.2,1.2,5.4,3.4,2.3,4.2]]
    # trainset, trainlabel = LSTM_rawset2trainset(rawset, 2)
    #
    # # 将ndarray转为tensor
    # trainset = torch.from_numpy(trainset)
    # trainlabel = torch.from_numpy(trainlabel)
    #
    # # 将tensor的后两个维度进行交换，由[N, dim, SEQ_LENGTH]形式变成[N, SEQ_LENGTH, dim]形式
    # trainset = trainset.transpose(1,2)
    # trainlabel = trainlabel.transpose(1,2)
    #
    # # 构建LSTM模型
    # lstm = LSTM(num_of_factors=5).double()
    #
    # # 测试模型有无错误
    # out = lstm.forward(trainset)
    # print(out.shape)
    #
    # # 训练LSTM模型
    # lstm = LSTM_train(lstm, trainset, trainlabel)
    #
    # # 利用LSTM模型进行预测
    # label = LSTM_predict(lstm, rawset)
    # print(label)
    print(final_lstm_predict(rawset))