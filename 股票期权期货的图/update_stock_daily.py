from lstm_predict import lstm_pred
from get_stock_info import portfolio_history_vol
from CNN import final_cnn_predict
from LSTM import final_lstm_predict
import numpy as np
import sqlite3
import math
import random
stock_pool=[]

start = '2018-01-02'
end = '2019-06-12'

def calc_pred_stock_return(stock_code, date):
        """
        stock_code是形如 SH600000 的字符串
        date是形如 2019-06-01 的字符串
        预测date日期后一天stock_code股票的收益率
        """
        conn = sqlite3.connect('fin_set.db')
        cursor = conn.cursor()
        string=[]
        for i in range(1,520):
            string.append('F'+str(i))
        for i in range(1,520):
            string.append('B'+str(i))
        cmd = ','.join(string)
        result = cursor.execute(
            'select  {}  from {} where date <= {}'.format(cmd,stock_code,'"' + date + '"'))
        count = 0
        factor=[[] for i in range(519)]
        beta=[0 for i in range(519)]
        for row in result:
            for i in range(519):
                factor[i].append(row[i])
            if count == 0:
                for j in range(519):
                    beta[j] = row[519+j]
            count += 1
            if count == 40:
                break

        result = cursor.execute('select RF from MARKET where date = {}'.format('"' + date + '"'))
        rf = 0
        for row in result:
            rf = row[0]
            break
        factor1=[]
        for i in range(5):
            factor1.append(factor[i])

        try:
            pf1=final_lstm_predict(factor1)
            pf2=final_cnn_predict(factor1)

            est_r1=rf
            est_r2=rf

            for i in range(5):
                est_r1+=pf1[i]*beta[i]
                est_r2+=pf2[i]*beta[i]

            for i in range(5,519):
                est_r1 += factor[i][-1] * beta[i]
                est_r2 += factor[i][-1] * beta[i]
        except:
            est_r1 = np.random.normal(loc=rf, scale=0.03, size=None)
            est_r2 = np.random.normal(loc=rf, scale=0.03, size=None)

        if math.fabs(est_r1)>0.1:
            sign = 1 if (random.random() > 0.5 ) else -1
            est_r1 = sign*0.1*random.random()

        if math.fabs(est_r2)>0.1:
            sign = 1 if (random.random() > 0.5 ) else -1
            est_r2 = sign*0.1*random.random()

        cursor.execute('update ESTSTK set EST_R = {} where TRADECODE = {} and date = {}'.format(est_r1,stock_code,'"' + date + '"',))
        cursor.execute('update ESTSTK set EST_R_2 = {} where TRADECODE = {} and date = {}'.format(est_r2, stock_code,
                                                                                                '"' + date + '"', ))
        conn.commit()
        conn.close()


def calc_pred_stock_vol(stock_code,date):
    conn = sqlite3.connect('fin_set.db')
    cursor = conn.cursor()
    result = cursor.execute(
        'select date,close from {} where date <={} order by date desc'.format(stock_code, '"' + date + '"'))
    count = 0
    dates_prices = []
    for row in result:
        dates_prices.append(row)
        count += 1
        if count == 31:
            break

    start_date = dates_prices[-1][0]
    portfolio=[stock_code]
    shares=[1]
    d,history_vols = portfolio_history_vol(portfolio,shares,start_date,date)
    ls = lstm_pred(history_vols)

    pred_vol = ls.predict()
    cursor.execute('update ESTSTK set EST_R = {} where TRADECODE = {} and date = {}'.format(pred_vol,stock_code,'"' + date + '"',))
    conn.commit()
    conn.close()

def update_database_daily(stock_pool,date):
    for stock_code in stock_pool:
        calc_pred_stock_vol(stock_code,date)
        calc_pred_stock_return(stock_code,date)



if __name__ == '__main__':
    conn = sqlite3.connect('finance_set.db')
    cursor = conn.cursor()
    # result = cursor.execute('select name from sqlite_master where type="table" order by name')
    # all = []
    # for row in result:
    #     all.append(row[0])
    #
    # for stock_code in all:
    #     result = cursor.execute('select * from {} where date = {} or date = {}'.format(stock_code,'"'+start+'"','"'+end+'"'))
    #     count = 0
    #     for row in result:
    #         count+=1
    #     if count == 2:
    #         stock_pool.append(stock_code+'\n')
    #
    # print(stoc k_pool)
    # conn.close()
    # f=open('stock_pool.txt','w')
    # f.writelines(stock_pool)
    # f.close()
    res = cursor.execute('select * from SH600717 where date = ? ',(start,))
    for row in res:
        print(row)
