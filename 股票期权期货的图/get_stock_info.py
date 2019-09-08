import sqlite3
import numpy as np
from lstm_predict import lstm_pred

def get_close_price(stock_code,start_date,end_date):
    conn = sqlite3.connect('finance_set.db')
    cursor = conn.cursor()
    result = cursor.execute(
        'select date,close from {} where date>={} and date <={}'.format(stock_code, '"' + start_date + '"',
                                                                        '"' + end_date + '"'))
    dates_prices = []
    for row in result:
        dates_prices.append(row)
    conn.close()
    return dates_prices

def get_N_days_close(stock_code,end_date,day_num=10):
    conn = sqlite3.connect('finance_set.db')
    cursor = conn.cursor()
    result = cursor.execute('select date,close from {} where date <={} order by date desc'.format(stock_code, '"' + end_date + '"'))
    count = 0
    dates_prices=[]
    for row in result:
        dates_prices.append(row)
        count+=1
        if count == day_num:
            break
    conn.close()
    return list(reversed(dates_prices))

def portfolio_history_return(portfolio,shares,start_date,end_date):
    """
        计算portfolio从start_date到end_date的历史收益率 不包括start_date
        start_date和end_date是形如 2019-06-01的字符串
    """
    prices = []
    for stock_code in portfolio:
        prices.append(get_close_price(stock_code,start_date,end_date))
    total=[]
    dates=[]
    for item in prices[0]:
        dates.append(item[0])

    for i in range(len(dates)):
        sum = 0
        for j in range(len(shares)):
            sum += shares[j]*prices[j][i][1]
        total.append(sum)

    returns=[]
    for i in range(1,len(total)):
        pre = total[i-1]
        cur = total[i]
        returns.append((cur-pre)/pre)

    return dates[1:],returns

def portfolio_history_vol(portfolio,shares,start_date,end_date):
    """
    计算portfolio从start_date到end_date的历史收益率的波动率 不包括start_date
    """
    prices=[]
    for stock_code in portfolio:
        prices1=get_N_days_close(stock_code,start_date,10)[:-1]
        prices1.extend(get_close_price(stock_code,start_date,end_date))
        prices.append(prices1)

    total=[]
    dates=[]
    for item in prices[0]:
        dates.append(item[0])

    for i in range(len(dates)):
        sum = 0
        for j in range(len(shares)):
            sum += shares[j]*prices[j][i][1]
        total.append(sum)

    returns = []
    for i in range(1, len(total)):
        pre = total[i - 1]
        cur = total[i]
        returns.append((cur - pre) / pre)

    vols=[]
    for i in range(9,len(returns)):
        vol=np.nanstd(np.array(returns[i-9:i+1]))
        vols.append(vol)
    return dates[10:],vols


def pred_portfolio_var(portfolio,shares,date):
    """
        计算portfolio在date的VaR
    """
    prices=[]
    for stock_code in portfolio:
        prices.append(get_N_days_close(stock_code,date))

    total = []
    dates = []
    for item in prices[0]:
        dates.append(item[0])

    for i in range(len(dates)):
        sum = 0
        for j in range(len(shares)):
            sum += shares[j] * prices[j][i][1]
        total.append(sum)

    returns = []
    for i in range(1, len(total)):
        pre = total[i - 1]
        cur = total[i]
        returns.append(np.log(cur/pre))

    sigma = np.std(returns)
    percentile95 = 1.6499
    return95 = np.exp(returns[-1] - percentile95 * sigma)
    return total[-1] - total[-1]*return95

def pred_stock_vol(stock_code,date):
    conn = sqlite3.connect('finance_set.db')
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
    conn.close()
    start_date = dates_prices[-1][0]
    portfolio=[stock_code]
    shares=[1]
    d,history_vols = portfolio_history_vol(portfolio,shares,start_date,date)
    ls = lstm_pred(history_vols)
    pred_vol = ls.predict()
    return pred_vol

if __name__=='__main__':
    #内部测试
    dp = get_close_price('SH600717','2019-02-01','2019-03-01')
    print(dp)
    dp2 = get_N_days_close('SH600000','2019-03-01',20)
    portfolio=['SH600000','SH600717']
    shares=[100,200]
    print(portfolio_history_return(portfolio,shares,'2019-02-01','2019-03-01'))
    dates,vols=portfolio_history_vol(portfolio,shares,'2019-02-01','2019-03-01')
    print(pred_portfolio_var(portfolio,shares,'2015-07-01'))
    print(pred_stock_vol('SH600717','2019-04-01'))

