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
    下面的函数用于将数据集分割为训练集和测试集
"""
def train_test_split(data, SEQ_LENGTH = 8, test_prop=0.15):
    ntrain = int(len(data) *(1-test_prop))   
    predictors = data.columns[:4]          # open, high, close, low
    data_pred = data[predictors]
    num_attr = data_pred.shape[1]          # 4
    
    result = np.empty((len(data) - SEQ_LENGTH, SEQ_LENGTH, num_attr))
    y = np.empty((len(data) - SEQ_LENGTH, SEQ_LENGTH))
    yopen = np.empty((len(data) - SEQ_LENGTH, SEQ_LENGTH))

    for index in range(len(data) - SEQ_LENGTH):
        result[index, :, :] = data_pred[index: index + SEQ_LENGTH]
        y[index, :] = data_pred[index+1: index + SEQ_LENGTH + 1].close

    """
        xtrain的大小：ntrain x SEQ_LENGTH x 4
        ytrain的大小：ntrain x SEQ_LENGTH
        
        * xtrain的每个batch为长为SEQ_LENGTH的连续序列，一共有ntrain个batch，
          序列中每个单元都是一个四元组（open，high，close，low）
        * ytrain的每个batch为长为SEQ_LENGTH的连续序列，一共有ntrain个batch，
          序列中每个单元是xtrain中对应四元组所在日期的下一天的close price
        
        xtest 的大小：    ntest x SEQ_LENGTH x 4                
        ytest的大小：     ntest x SEQ_LENGTH      (close price)
        ytest_open的大小：ntest x SEQ_LENGTH      (open price)  
        
        * xtest的每个batch为长为SEQ_LENGTH的连续序列，一共有ntest个batch，
          序列中每个单元都是一个四元组（open，high，close，low）
          每一个序列仅包含一个新四元组，且在最后一个
        * ytest的每个batch为长为SEQ_LENGTH的连续序列，一共有ntest个batch，
          序列中每个单元是xtest中对应四元组所在日期的下一天的close price
        
        类型：numpy.ndarray
    """
    xtrain = result[:ntrain, :, :]
    ytrain = y[:ntrain]
    
    xtest = result[ntrain:, :, :]
    ytest = y[ntrain:]
    
    return xtrain, xtest, ytrain, ytest
    
     
"""
    LSTM模型的定义
    
    模型参数：input_dim : 每一刻输入的维度
              output_dim: 输出的维度      (默认为1，即每个LSTM模型专门预测某一个指标)
              hidden_dim: 隐层神经元个数，越大则模型复杂度越高
              num_layers: LSTM中lstm cell个数，越大则模型复杂度越高
              
    模型输入：[N, SEQ_LENGTH, input_dim] (N为样本个数)
    模型输出：[N, SEQ_LENGTH]
    (输入输出的含义可见train_test_split函数)
    
    这里使用LSTM模型预测未来一天的收盘价
"""
class LSTM(nn.Module):

    def __init__(self, input_dim=4, hidden_dim=15, batch_size=1, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

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
    LSTM模型的测试函数(或称inference)
"""
def LSTM_test(model, xtest, ytest):
    # pred_test为预测结果
    with torch.no_grad():
        pred_test  = lstm(xtest)

    plt.plot(np.array(ytest[:,-1]), label='true values')
    plt.plot(np.array(pred_test[:,-1]), label='predicted  values')
    plt.legend()
    plt.title('closed price')
    plt.show()
    
 
        
if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv('stock_data_730.csv')
    data.set_index(["date"], inplace=True)
    data_sorted = data.sort_index()
    
    # 划分训练集与测试集并转为tensor类型
    xtrain, xtest, ytrain, ytest = train_test_split(data_sorted)
    
    xtrain = torch.from_numpy(xtrain)
    ytrain = torch.from_numpy(ytrain)
    xtest = torch.from_numpy(xtest)
    ytest = torch.from_numpy(ytest)
    
    # 构建LSTM模型
    lstm = LSTM().double()
    
    # 训练LSTM模型
    lstm = LSTM_train(lstm, xtrain, ytrain)
    
    # 测试LSTM模型
    LSTM_test(lstm, xtest, ytest)
    