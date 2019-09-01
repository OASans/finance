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
    CNN模型的定义
    该CNN模型包含两个卷积层以及三个全连接层，可以由前几天的数据预测未来的收益率
    
    目前该网络的输入为[N, 1, 8, 4] N为样本数，1为通道个数，8和4分别为图片的长和宽
    如果想要输入不同长宽的图片，想要修改网络参数
    网络的输出为[N, 1]，
""" 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(768, 128),
            nn.PReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.PReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, 1),         # 直接预测收益率
            nn.PReLU(),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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
    for i in range(50):
        print('STEP: ', i)
    
        def closure():
            optimizer.zero_grad()
            out = cnn(xtrain_cnn)
            loss = criterion(out, ytrain_cnn)
            print('loss: %5.3f  ' %(loss.item()))
            loss.backward()
        
            return loss
    
        optimizer.step(closure)
       
    return model
    
    
"""
    CNN模型的测试函数
"""
def CNN_test(model, xtest, ytest):
    # pred_yieldrate为预测的收益率
    with torch.no_grad():
        pred_yieldrate  = cnn(xtest_cnn)
    
    plt.plot(ytest, label='true values')
    plt.plot(np.array(pred_yieldrate), label='predicted  values')
    plt.legend()
    plt.title('yield rate')
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
    
    # 将原矩阵变成单通道的图片
    xtrain_cnn = xtrain.unsqueeze(1).float()
    xtest_cnn = xtest.unsqueeze(1).float()
    
    # 使用cnn直接预测收益率
    ytrain_np = np.array(ytrain)
    ytest_np = np.array(ytest)
    
    ytrain_cnn = (ytrain_np[:, -1] - ytrain_np[:, 0]) / ytrain_np[:, 0]
    ytest_yieldrate = (ytest_np[:, -1] - ytest_np[:, 0]) / ytest_np[:, 0]
    
    ytrain_cnn = torch.from_numpy(ytrain_cnn).unsqueeze(1).float()
    
    # 构建CNN模型
    cnn = CNN()
    
    # 训练CNN模型
    cnn = CNN_train(cnn, xtrain_cnn, ytrain_cnn)
    
    # 测试CNN模型
    CNN_test(cnn, xtest_cnn, ytest_yieldrate)