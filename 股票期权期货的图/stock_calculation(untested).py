from FetchData import FetchData
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=15, batch_size=1, output_dim=1, num_layers=2):
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
            y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
            outputs.append(y_pred)

        return torch.stack(outputs).squeeze(1).squeeze(1)

class History_analysis(object):
    def __init__(self, history_price, shares):
        """
        history_price数据类型是ndarray,资产历史价格 m*n维 m是资产个数 n是天数
        该类可以整合个股和组合的功能 个股只需m=1即可 之后调用时name均为None
        shares数据类型是ndarray，资产份额 长度为m
        能用numpy作计算就numpy 提高速度
        """
        self.history_price = history_price
        self.shares = shares
        self.SEQ_LENGTH = 8
        #未计算 先设为None
        self.daily_return = None
        self.total_asset = None
        self.vol = None
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.model = None

    def fill_nan(self):
        "填充nan（停牌或未上市）数值 用前一天的价格表示当天价格 如果第一天就是nan 用第一个不是nan的数值填充第一天"
        m,n=self.history_price.shape
        for i in range(m):
            local = self.history_price[i]
            nan = np.isnan(local)
            if nan[0]==True:
                for j in range(1,n):
                    if nan[j]==False:
                        local[0]=local[j]
                        break
            for j in range(1,n):
                if nan[j] == True:
                    local[j]=local[j-1]

    def cal_history_return(self):
        "先填充nan 再做计算每日收益率"
        num,length=self.history_price.shape
        daily_return = np.zeros(length - 1)

        self.total_asset=self.shares.dot(self.history_price)

        for i in range(1,length):
            daily_return[i] = (self.total_asset[i]-self.total_asset[i-1])/self.total_asset[i-1]

        self.daily_return = daily_return
        return daily_return

    #计算波动率 以年为单位
    def volatility(sequence):
        yield_rates = [(sequence[i + 1] - sequence[i]) / sequence[i] for i in range(len(sequence) - 1)]
        return np.std(yield_rates) * np.sqrt(250)

    #计算收益的波动率
    def cal_volatility(self):
        length = self.total_asset.shape[0]
        self.vol=np.zeros(length-self.SEQ_LENGTH+1)
        for i in range(length-self.SEQ_LENGTH+1):
            self.vol = self.volatility(self.total_asset[i:i+self.SEQ_LENGTH])
        return self.vol


    def train_test_split(self,data, test_prop=0.0):  # N=7, SEQ_LENGTH=N+1
        """
            data type:   np.ndarray
            return type: np.ndarray
            test_prop设为0.0 暂时不需要测试集 全部过拟合即可
        """
        xtrain, ytrain = [], []
        xtest, ytest = [], []
        for i in range(0, int(len(data) * (1 - test_prop)) - self.SEQ_LENGTH):
            #用SEQ_LENGTH-1的历史 预测后一天
            seq = data[i:i + self.SEQ_LENGTH]
            xtrain.append(seq[:-1])
            ytrain.append(seq[-1])

        for i in range(int(len(data) * (1 - test_prop)) - self.SEQ_LENGTH, len(data) - self.SEQ_LENGTH):
            seq = data[i:i + self.SEQ_LENGTH]
            xtest.append(seq[:-1])
            ytest.append(seq[-1])

        self.xtrain=np.array(xtrain)
        self.ytrain=np.array(ytrain)
        self.xtest=np.array(xtest)
        self.ytest=np.array(ytest)

    def train_LSTM(self):
        "训练LSTM模型"
        self.train_test_split(self.total_asset)
        self.model = LSTM().double()
        criterion = nn.MSELoss()
        # 优化器
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.1)
        # 开始训练 40可以改
        for i in range(40):
            def closure():
                optimizer.zero_grad()
                out = self.model(self.xtrain)
                loss = criterion(out, self.ytrain)
                print('loss: %5.3f  ' % (loss.item()))
                loss.backward()
                return loss
            optimizer.step(closure)


    def predict_return(self):
        last_N_x=self.total_asset[-(self.SEQ_LENGTH-1):]
        y_pred = self.model(last_N_x)[0].item()
        pred_return = (y_pred - self.total_asset[-1])/self.total_asset[-1]
        return pred_return

    def predict_vol(self):
        last_N_x = self.total_asset[-(self.SEQ_LENGTH - 1):]
        y_pred = self.model(last_N_x)[0].item()
        pred_vol = self.volatility(np.append(last_N_x,y_pred))
        return pred_vol


    def VaR_calculate(self):
        last_N_x = self.total_asset[-(self.SEQ_LENGTH - 1):]
        pred_return = self.predict_return()
        all = np.append(last_N_x,pred_return)
        log_return = np.zeros(self.SEQ_LENGTH-1)
        for i in range(len(log_return)):
            log_return[i]=np.log(all[i+1]/all[i])
        log_vol=np.std(log_return)
        log_pred_return = np.log(pred_return)
        percentile95 = 1.6499
        var95 = - np.exp(log_pred_return-percentile95*log_vol)
        return var95*self.total_asset[-1]

if __name__=='__main__':
    #从fetchdata里获得股票组合的historydata
    #history ndarray
    #shares ndarray
    history=None
    shares=None

    analysis = History_analysis(history,shares)
    analysis.fill_nan() #填充nan值 规则见函数

    daily_return = analysis.cal_history_return() #计算每日收益率 （和每日总资产 不返回）
    volatility = analysis.cal_volatility() #计算每日波动率 长度比history短SEQ_LENGTH-1

    analysis.train_test_split()
    analysis.train_LSTM() #预测总资产数量的模型

    pred_return = analysis.predict_return()
    pred_vol = analysis.predict_vol()

    var95 = analysis.VaR_calculate() #95%的VaR