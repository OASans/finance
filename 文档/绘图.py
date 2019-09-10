def pred_stock_return(stock_code,date):
    """
    stock_code是形如 SH600000 的字符串 下同
    date是形如 2019-06-01 的字符串 下同
    预测date日期后一天stock_code股票的收益率
    """
    pred=1.0 #pred -> float
    return pred

def pred_stock_vol(stock_code,date):
    """
    预测date日期后一天stock_code股票的收益率的波动率
    """
    pred=1.0 #pred -> float
    return pred

def pred_portfolio_return(portfolio,shares,date):
    """
    portfolio是包含stock_code的列表,如果是单只股票则只包含一个stock_code即可 下同
    shares是每个股票数量的列表与stock_code一一对应 下同
    预测该portfolio在date之后一日的收益率
    """
    pred=1.0 #pred -> float
    return pred

def pred_portfolio_var(portfolio,shares,date):
    """
    计算portfolio在date的VaR
    """
    pred=1.0 #pred -> float
    return pred

def portfolio_history_return(portfolio,shares,start_date,end_date):
    """
    计算portfolio从start_date到end_date的历史收益率
    start_date和end_date是形如 2019-06-01的字符串
    """
    dates=['2019-06-01','2019-06-02','2019-06-03']
    returns=[0.1,0.1,0.1] 
    return dates,returns #返回类型为列表

def portfolio_history_vol(portfolio,shares,start_date,end_date):
    """
    计算portfolio从start_date到end_date的历史收益率的波动率
    """
    dates=['2019-06-01','2019-06-02','2019-06-03']
    vols=[0.1,0.1,0.1] 
    return dates,vols