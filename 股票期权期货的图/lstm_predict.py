import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import random
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

class lstm_pred():
    def __init__(self,data):
        for i in range(len(data)):
            data[i]+=0.00001*random.random()
        self.data = np.array(data)
        self.SEQ_LENGTH = 8
        self.xtrain = []
        self.ytrain = []
        self.model = LSTM().double()

    def train_test_split(self, test_prop=0.0):  # N=7, SEQ_LENGTH=N+1
        """
            data type:   np.ndarray
            return type: np.ndarray
        """

        xtrain, ytrain = [], []
        xtest, ytest = [], []

        for i in range(0, int(len(self.data) * (1 - test_prop)) - self.SEQ_LENGTH):
            seq = self.data[i:i + self.SEQ_LENGTH]
            # yield_rates = [(seq[i+1]-seq[i])/seq[i] for i in range(SEQ_LENGTH-1)]
            # xtrain.append(yield_rates[:-1])                                         # 收益率
            # ytrain.append(np.std(yield_rates) * np.sqrt(250))                       # 历史波动率
            xtrain.append(seq[:-1])
            ytrain.append(seq[-1])

        for i in range(int(len(self.data) * (1 - test_prop)) - self.SEQ_LENGTH, len(self.data) - self.SEQ_LENGTH):
            seq = self.data[i:i + self.SEQ_LENGTH]
            # yield_rates = [(seq[i+1]-seq[i])/seq[i] for i in range(SEQ_LENGTH-1)]
            # xtest.append(yield_rates[:-1])
            # ytest.append(np.std(yield_rates) * np.sqrt(250))
            xtest.append(seq[:-1])
            ytest.append(seq[-1])

        self.xtrain=torch.from_numpy(np.array(xtrain))
        self.ytrain=torch.from_numpy(np.array(ytrain))

    def train_model(self):
        criterion = nn.MSELoss()

        # 优化器
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.1)

        # 开始训练
        for i in range(10):
            #print('STEP: ', i)

            def closure():
                optimizer.zero_grad()
                out = self.model(self.xtrain)
                loss = criterion(out, self.ytrain)
                #print('loss: %5.3f  ' % (loss.item()))
                loss.backward()
                return loss
            optimizer.step(closure)

    def predict(self):
        self.train_test_split()
        self.train_model()
        feed = torch.from_numpy(np.array([self.data[-self.SEQ_LENGTH+1:]]))
        #print(self.xtrain)
        #print(feed)
        pred = self.model(feed)
        return float(pred[0].item())


if __name__=='__main__':
    data = [13.47,13.45,13.5,13.71,14.26,14.4,14.49,15.18,15.36,15.59,15.43,15.31,15.26,15.26,15.18,15.07,14.9,14.69,15.25,14.89,14.64,14.73,14.96,14.4,15.04,15.38,16.41,16.91,16.54,16.27]
    data2 = [10 for i in range(30)]
    time1=time.time()
    ls = lstm_pred(data2)
    c = ls.predict()
    print(c)
    time2=time.time()
    print(time2-time1)





