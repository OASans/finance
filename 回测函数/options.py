# -*- coding: utf-8 -*-
# author Yuan Manjie
# date 2019/9/5

import pandas as pd
import numpy as np
import sqlite3

conn = sqlite3.connect('../获取资产的基本数据/fin_set.db')#连接到db
c = conn.cursor()#创建游标

pd.set_option('mode.use_inf_as_na', True)

contract_unit=dict()
contract_unit['50ETF']=10000
contract_unit['IF']=300
contract_unit['IC']=200
contract_unit['IH']=300

# stock_path = r'../获取资产的基本数据/股票/'
# options_path = r'../获取资产的基本数据/期权/'
# futures_path = r'../获取资产的基本数据/期货/'

class ID_Not_Exist_ERROR(Exception):
    def __init__(self,id):
        self.id=id

# 获取根据id获取一段时期内证券的数据
# @auto-fold here
def format_date(date):
    pd_t=pd.Timestamp(date)
    return str(pd_t)[:10]

# @auto-fold here
def get_stock_data(id, begin_t='', end_t='',t_before=0):
    new_id=id[-2:]+id[:6]
    try:
        if begin_t!=''and end_t!='':
            b_t=format_date(begin_t)
            e_t=format_date(end_t)
            sql="select \"index\" from "+new_id+" where DATE>='"+b_t+"' and DATE<='"+e_t+"'"
            sql_dat=list(c.execute(sql))
            b_i=sql_dat[0][0]
            e_i=sql_dat[-1][0]
            sql="select DATE,OPEN from "+new_id+" where \"index\">='"+str(b_i-t_before)+"' and \"index\"<='"+str(e_i)+"'"
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['date','open'])
            sql_dat.index=map(pd.Timestamp,sql_dat['date'])
        else:
            sql="select DATE,OPEN from "+new_id
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['date','open'])
            sql_dat.index=map(pd.Timestamp,sql_dat['date'])

        # data = pd.read_excel(stock_path + id + r'.xlsx',index_col=0)
        # if begin_t!=''and end_t!='':
            # data=data[begin_t:end_t]
        return sql_dat.iloc[:,1:]
    except :
        raise ID_Not_Exist_ERROR(id)

# @auto-fold here
def get_futures_data(id, begin_t='', end_t='',t_before=0):
    new_id=id
    try:
        if type(begin_t)==type(1):
            sql="select \"index\" from "+new_id
            sql_dat=list(c.execute(sql))
            b_i=sql_dat[begin_t][0]
            e_i=sql_dat[end_t][0]
            sql="select * from "+new_id+" where \"index\">='"+str(b_i-t_before)+"' and \"index\"<='"+str(e_i)+"'"
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','DATE','PRESETTLE','PRECLO','OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','VWAP','OI_CHG','DELTA','GAMMA','VEGA','THETA','RHO','VOLATILITYRATIO','US_IMPLIEDVOL'])
            sql_dat.index=map(pd.Timestamp,sql_dat['DATE'])
        if begin_t!=''and end_t!='':
            b_t=format_date(begin_t)
            e_t=format_date(end_t)
            sql="select \"index\" from "+new_id+" where DATE>='"+b_t+"' and DATE<='"+e_t+"'"
            sql_dat=list(c.execute(sql))
            b_i=sql_dat[0][0]
            e_i=sql_dat[-1][0]
            sql="select * from "+new_id+" where \"index\">='"+str(b_i-t_before)+"' and \"index\"<='"+str(e_i)+"'"
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','DATE','PRESETTLE','PRECLO','OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','OI_CHG','VWAP','VOLRATIO'])
            sql_dat.index=map(pd.Timestamp,sql_dat['DATE'])
        else:
            sql="select * from "+new_id
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','DATE','PRESETTLE','PRECLO','OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','OI_CHG','VWAP','VOLRATIO'])
            sql_dat.index=map(pd.Timestamp,sql_dat['DATE'])
        # data = pd.read_excel(futures_path + id + r'.xlsx',index_col=0)
        # # data = pd.read_csv(futures_path + id + r'.csv',index_col=0,engine='python')
        # if begin_t!=''and end_t!='':
        #     data=data[begin_t:end_t]
        return sql_dat
    except :
        raise ID_Not_Exist_ERROR(id)

# @auto-fold here
def get_options_data(id, begin_t='', end_t='',t_before=0):
    new_id=id[-2:]+id[:8]
    try:
        if type(begin_t)==type(1):
            sql="select \"index\" from "+new_id
            sql_dat=list(c.execute(sql))
            b_i=sql_dat[begin_t][0]
            e_i=sql_dat[end_t][0]
            sql="select * from "+new_id+" where \"index\">='"+str(b_i-t_before)+"' and \"index\"<='"+str(e_i)+"'"
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','DATE','PRESETTLE','PRECLO','OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','VWAP','OI_CHG','DELTA','GAMMA','VEGA','THETA','RHO','VOLATILITYRATIO','US_IMPLIEDVOL'])
            sql_dat.index=map(pd.Timestamp,sql_dat['DATE'])
        elif begin_t!=''and end_t!='':
            b_t=format_date(begin_t)
            e_t=format_date(end_t)
            sql="select \"index\" from "+new_id+" where DATE>='"+b_t+"' and DATE<='"+e_t+"'"
            sql_dat=list(c.execute(sql))
            b_i=sql_dat[0][0]
            e_i=sql_dat[-1][0]
            sql="select * from "+new_id+" where \"index\">='"+str(b_i-t_before)+"' and \"index\"<='"+str(e_i)+"'"
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','DATE','PRESETTLE','PRECLO','OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','VWAP','OI_CHG','DELTA','GAMMA','VEGA','THETA','RHO','VOLATILITYRATIO','US_IMPLIEDVOL'])
            sql_dat.index=map(pd.Timestamp,sql_dat['DATE'])
        else:
            sql="select * from "+new_id
            sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','DATE','PRESETTLE','PRECLO','OPEN','HIGH','LOW','CLOSE','SETTLE','VOLUME','AMT','OI','VWAP','OI_CHG','DELTA','GAMMA','VEGA','THETA','RHO','VOLATILITYRATIO','US_IMPLIEDVOL'])
            sql_dat.index=map(pd.Timestamp,sql_dat['DATE'])
        sql='select EXE_PRICE,ENDDATE from OPTIONINFO where TRADECODE=\''+id+'\''
        sql_tmp=list(c.execute(sql))
        sql_dat['EXE_PRICE']=sql_tmp[0][0]
        sql_dat['EXE_ENDDATE']=pd.Timestamp(sql_tmp[0][1])
        # data = pd.read_excel(options_path + id + r'.xlsx',index_col=0)
        # # data = pd.read_csv(options_path + id + r'.csv',index_col=0,engine='python')
        # if begin_t!=''and end_t!='':
        #     data=data[begin_t:end_t]
        return sql_dat
    except Exception as e:
        raise ID_Not_Exist_ERROR(id)

# @auto-fold here
def is_stock(id):
    if id[-3:] in ['.SZ','.SH'] and len(id)==9:
        return True
    else:
        return False

# @auto-fold here
def is_futures(id):
    if id[:2] in ['IF','IC','IH'] and len(id)==6:
        return True
    else:
        return False

# @auto-fold here
def is_options(id):
    if id[-3:] in ['.SH'] and len(id)==11:
        return True
    else:
        return False

# @auto-fold here
def portfolio_total_value(asset_id,asset_mount,cash,begin_t, end_t,t_before=0):
    total=[]
    stocks=[]
    keys=[]
    stock_keys=[]
    for ii,i in enumerate(asset_id):# 到期？ 保证金不足？
        if is_stock(i):
            temp=get_stock_data(i,begin_t,end_t,t_before)
            if len(temp)!=0:
                keys+=[i]
                stock_keys+=[i]
                total+=[temp*asset_mount[ii]]
                stocks+=[temp*asset_mount[ii]]
        elif is_futures(i):
            temp=get_futures_data(i, begin_t, end_t,t_before)
            if len(temp)!=0:
                keys+=[i]
                temp['delta']=temp['OPEN'].diff(1)
                temp=temp.fillna(0)
                temp['delta']*=contract_unit[i[:2]]
                temp['delta'][0]+=temp['OPEN'][0]*contract_unit[i[:2]]*0.08
                for j in range(1,len(temp)):
                    temp['delta'][j]+=temp['delta'][j-1]
                    if temp['delta'][j]<=0:
                        temp['delta'][j:]=0
                        break
                total+=[temp['delta']*asset_mount[ii]]
        elif is_options(i):
            temp=get_options_data(i,begin_t,end_t,t_before)
            temp=temp.fillna(0)
            if len(temp)!=0:
                keys+=[i]
                total+=[temp['OPEN']*asset_mount[ii]*contract_unit['50ETF']]
    if len(total)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total,axis=1,keys=keys)
        total=total.fillna(method='ffill')
        total=total.fillna(method='bfill')
        if len(stocks)==0:
            stocks=pd.DataFrame()
        else:
            stocks=pd.concat(stocks,axis=1,keys=stock_keys)
            stocks=stocks.fillna(method='ffill')
            stocks=stocks.fillna(method='bfill')
    return total.sum(axis=1)+cash,stocks.sum(axis=1)

# @auto-fold here
def portfolio_delta(asset_id,asset_mount,cash,begin_t, end_t):#单个值可能不好取   单位上？
    total,stock=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t,1)
    delta=total.diff(1)/stock.diff(1)
    return delta[1:]

# @auto-fold here
def portfolio_gamma(asset_id,asset_mount,cash,begin_t, end_t):
    total,stock=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t,2)
    delta=total.diff(1)/stock.diff(1)
    gamma=delta.diff(1)/stock.diff(1)
    return gamma[2:]

# @auto-fold here
def portfolio_vega(asset_id,asset_mount,cash,begin_t,end_t):
    total_vega=[]
    keys=[]
    for ii,i in enumerate(asset_id):
        if is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                total_vega+=[temp['VEGA']*asset_mount[ii]*contract_unit['50ETF']]
    if len(total_vega)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total_vega,axis=1,keys=keys)
        total=total.fillna(0)
        return total.sum(axis=1)

# @auto-fold here
def portfolio_rho(asset_id,asset_mount,cash,begin_t,end_t):
    total_rho=[]
    keys=[]
    for ii,i in enumerate(asset_id):
        if is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                total_rho+=[temp['RHO']*asset_mount[ii]*contract_unit['50ETF']]
        elif is_futures(i):
            pass
            # temp=get_futures_data(i, begin_t, end_t)
            # if len(temp)!=0:
            #     keys+=[i]
            #     temp['rho']=list(map(lambda x:1.015**(x.days/365)*x.days/365,temp['LASTTRADE_DATE']-temp.index))
            #     total_rho+=[temp['OPEN']*temp['rho']*asset_mount[ii]*contract_unit[i[:2]]]
    if len(total_rho)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total_rho,axis=1,keys=keys)
        total=total.fillna(0)
        return total.sum(axis=1)

# @auto-fold here
def portfolio_theta(asset_id,asset_mount,cash,begin_t,end_t):
    total_theta=[]
    keys=[]
    for ii,i in enumerate(asset_id):
        if is_options(i):
            temp=get_options_data(i,begin_t,end_t)
            if len(temp)!=0:
                keys+=[i]
                total_theta+=[temp['THETA']*asset_mount[ii]*contract_unit['50ETF']]
        elif is_futures(i):
            pass
            # temp=get_futures_data(i, begin_t, end_t)
            # if len(temp)!=0:
            #     keys+=[i]
            #     temp['theta']=list(map(lambda x:-np.log(1.015)*1.015**(x.days/365),temp['LASTTRADE_DATE']-temp.index))
            #     total_theta+=[temp['OPEN']*temp['theta']*asset_mount[ii]*contract_unit[i[:2]]]
    if len(total_theta)==0:
        return pd.DataFrame()
    else:
        total=pd.concat(total_theta,axis=1,keys=keys)
        total=total.fillna(0)
        return total.sum(axis=1)

# portfolio_var 调用这里portfolio_total_value取第一个返回值变为list调用定期调整与条件触发中的代码实现

# @auto-fold here
def portfolio_volatility(asset_id,asset_mount,cash,begin_t,end_t,time=10):
    total,_=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t,time)
    # total=np.log(total)
    res=total.diff(1)/total
    res=res.rolling(time).std()
    return res.dropna()

# @auto-fold here
def portfolio_earning_rate(asset_id,asset_mount,cash,begin_t,end_t,time=10):
    total,_=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    res=total.diff(1)/total
    res=res.rolling(time).mean()
    # res=res/time*365
    return res.dropna()

# @auto-fold here
def cal_option_amt(total_value,option,portion,t1):
    try:
        temp=get_options_data(option,t1,t1)
    except:
        return 0
    if len(temp)<=0:
        return 0
    else:
        res=total_value*portion/contract_unit['50ETF']/temp['EXE_PRICE'][-1]
        return int(res+0.5)

# @auto-fold here
def cal_future_amt(total_value,futures,portion,t1):
    try:
        temp=get_futures_data(futures,t1,t1,1)
    except:
        return 0
    if len(temp)<=0:
        return 0
    else:
        res=total_value*portion/contract_unit[futures[:2]]/temp['OPEN'][-1]
        return int(res+0.5)

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
# @auto-fold here
def load_train_data(asset_id,asset_mount,cash,options,begin_t='',end_t='',mode=0,train=0):
    # data = pd.read_excel(options_path + options + r'.xlsx',index_col=0)
    if train==1:
        data=get_options_data(options,begin_t,end_t,2)
    else:
        data=get_options_data(options,begin_t,end_t)
    s,_=portfolio_total_value(asset_id, asset_mount, cash, data.index[0], data.index[-1])
    data=pd.concat([data,s],axis=1)
    data.columns=list(data.columns[:-1])+['s']
    data=data[~np.isnan(data['s'])]

    data['ds']=data['s'].diff()
    data['f']=data['OPEN']*contract_unit['50ETF']
    data['df']=data['f'].diff()
    data['real_delta']=data['df']/data['ds']
    if mode==1:
        data['dds']=data['real_delta'].diff()
        data['real_gamma']=data['dds']/data['ds']

    if mode==1:
        data['gamma_pre']=data['GAMMA'].shift(-1)
        data['real_gamma']=data['real_gamma'].shift(-1)

    data['delta_pre']=data['DELTA'].shift(-1)
    data['real_delta']=data['real_delta'].shift(-1)#错开一天，即为预测下一天


    data['PCT_CHG']=data['df']/data['OPEN']
    data['HV_5']=(data['PCT_CHG'].rolling(window=5).std().values) #波动率
    data['HV_10']=(data['PCT_CHG'].rolling(window=10).std().values)
    data['HV_15']=(data['PCT_CHG'].rolling(window=15).std().values)
    data['HV_20']=(data['PCT_CHG'].rolling(window=20).std().values)

    data['SoverK']=data['s']/data['EXE_PRICE']
    data['T-t']=list(map(lambda x:x.days/365,data['EXE_ENDDATE']-data.index))

    for i in range(len(data)): #隐含波动率没有解的（为空的），用20日波动率替代
        if data['US_IMPLIEDVOL'][i]==0:
            data['US_IMPLIEDVOL'][i]=data['HV_20'][i]
    if train:
        data.dropna(inplace=True)
    else:
        data=data.where(data.notnull(),0)
    if train:
        data=data[abs(data['real_delta'])<=0.95] #选取delta在正负0.95以内的数据训练，删去极端值，原因是参考某某论文
    # if mode==1:
    #     pass
    # else:
    #     data['real_delta']=1/data['real_delta']

    if mode==1:
        data_train=data[['EXE_PRICE','s','T-t','US_IMPLIEDVOL','HV_5','HV_10','HV_15','HV_20','VOLATILITYRATIO','DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO','VWAP', 'VOLUME', 'AMT', 'OI_CHG', 'SETTLE','HIGH', 'LOW','delta_pre','real_delta','gamma_pre','real_gamma']]
    else:
        data_train=data[['EXE_PRICE','s','T-t','US_IMPLIEDVOL','HV_5','HV_10','HV_15','HV_20','VOLATILITYRATIO','DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO','VWAP', 'VOLUME', 'AMT', 'OI_CHG', 'SETTLE','HIGH', 'LOW','delta_pre','real_delta']]
    data_train.dropna(inplace=True)
    return data_train

def train_delta_model(protfolio_id,asset_id,asset_mount,cash,options,num=0):
    data_train=load_train_data(asset_id, asset_mount, cash, options,train=1)
    rfr = RandomForestRegressor()
    rfr.fit(data_train.iloc[:,:-2].values,data_train.iloc[:,-1].values)
    joblib.dump(rfr,str(protfolio_id)+"_delta"+str(num)+".m")

    # def sse(dataa,x):
    #     return np.sum(np.square(dataa['real_delta']-x))#或是real_delta
    # def test(dataa,x):
    #     p=np.array(x.predict(dataa.iloc[:,:-2].values))
    #     sse1=sse(dataa,p)
    #     sse2=sse(dataa,dataa['delta_pre'])#或是delta_pre
    #     res=1-sse1/sse2
    #     print(sse1,sse2)
    #     print(res)
    # test(data_train,rfr)

    return rfr

def retrain_delta_model(protfolio_id,asset_id,asset_mount,cash,options,num=0):
    return train_delta_model(protfolio_id, asset_id, asset_mount, cash, options,num=num)

def train_gamma_model(protfolio_id,asset_id,asset_mount,cash,options,num):
    data_train=load_train_data(asset_id, asset_mount, cash, options,mode=1,train=1)
    rfr = RandomForestRegressor()
    rfr.fit(data_train.iloc[:,:-4].values,data_train.iloc[:,-1].values)
    joblib.dump(rfr,str(protfolio_id)+"_gamma"+str(num)+".m")
    return rfr

def retrain_gamma_model(protfolio_id,asset_id,asset_mount,cash,options1,options2):
    model1=train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options1,1)
    model2=train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options2,2)
    model3=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options1,1)
    model4=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options2,2)


def fit_delta(protfolio_id,asset_id,asset_mount,cash,options,begin_t,end_t):
    try:
        model=joblib.load(str(protfolio_id)+"_delta0.m")
    except:
        model=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options)
    data=load_train_data(asset_id,asset_mount,cash,options,begin_t,end_t,train=0)
    res=model.predict(data.iloc[:,:-2].values)
    res=list(map(lambda x:0 if x>=0 else -1/x,res))
    data['pred']=res
    return data['pred']

def fit_gamma(protfolio_id,asset_id,asset_mount,cash,options1,options2,begin_t,end_t):
    try:
        model1=joblib.load(str(protfolio_id)+"_gamma1.m")
    except:
        model1=train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options1,1)
    try:
        model2=joblib.load(str(protfolio_id)+"_gamma2.m")
    except:
        model2=train_gamma_model(protfolio_id, asset_id, asset_mount, cash, options2,2)
    try:
        model3=joblib.load(str(protfolio_id)+"_delta1.m")
    except:
        model3=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options1,1)
    try:
        model4=joblib.load(str(protfolio_id)+"_delta2.m")
    except:
        model4=train_delta_model(protfolio_id, asset_id, asset_mount, cash, options2,2)

    data1=load_train_data(asset_id,asset_mount,cash,options1,begin_t,end_t,mode=1,train=0)
    data2=load_train_data(asset_id,asset_mount,cash,options2,begin_t,end_t,mode=1,train=0)
    data=pd.concat([data1,data2],axis=1)
    data=data.fillna(method='ffill')
    data=data.fillna(method='bfill')
    gamma1=model1.predict(data.iloc[:,:21].values)
    gamma2=model2.predict(data.iloc[:,25:46].values)
    delta1=model3.predict(data.iloc[:,:21].values)
    delta2=model4.predict(data.iloc[:,25:46].values)
    gamma1=pd.Series(map(lambda x:0 if x ==0 else x,gamma1))
    gamma2=pd.Series(map(lambda x:0 if x ==0 else x,gamma2))
    delta1=pd.Series(map(lambda x:0 if x ==0 else x,delta1))
    delta2=pd.Series(map(lambda x:0 if x ==0 else x,delta2))
    temp=gamma1*delta2-gamma2*delta1
    x1=gamma2/temp
    x2=gamma1/(-temp)
    x1=x1.map(lambda x:0 if x <=0 else x)
    x2=x2.map(lambda x:0 if x <=0 else x)
    data['x1']=list(x1)
    data['x2']=list(x2)
    return data[['x1','x2']]

from sklearn import linear_model
def load_train_future_data(asset_id,asset_mount,cash,future,begin_t='',end_t=''):
    data = get_futures_data(future, begin_t, end_t)['OPEN']
    s,_=portfolio_total_value(asset_id, asset_mount, cash, begin_t, end_t)
    data=pd.concat([data,s],axis=1,keys=['future','s'])
    data['r_f']=data['future'].diff()/data['future']
    data['r_s']=data['s'].diff()/data['s']

    data=data[~np.isnan(data['s'])]
    data=data.dropna()
    return data[['r_f','r_s']]

def retrain_beta_model(protfolio_id,asset_id,asset_mount,cash,futures,num=0):
    return train_beta_model(protfolio_id, asset_id, asset_mount, cash, futures,num=num)

def train_beta_model(protfolio_id,asset_id,asset_mount,cash,futures,num=0):
    data_train=load_train_future_data(asset_id, asset_mount, cash, futures)
    model = linear_model.LinearRegression()
    model.fit(data_train.iloc[:,0:1].values,data_train.iloc[:,1].values)
    joblib.dump(model,str(protfolio_id)+"_beta"+str(num)+".m")
    return model

def fit_beta(protfolio_id,asset_id,asset_mount,cash,futures):
    try:
        model=joblib.load(str(protfolio_id)+"_beta0.m")
    except:
        model=train_beta_model(protfolio_id, asset_id, asset_mount, cash, futures)
    res=model.coef_
    return res[0]


def generate_recommend_option_delta(protfolio_id,asset_id,asset_mount,cash):
    sql="select * from OPTIONINFO "
    sql_dat=pd.DataFrame(list(c.execute(sql)),columns=['index','TRADECODE','EXE_PRICE','EXE_MODE','FIRST_DATE','LAST_DATE'])
    sql="select DATE from "+sql_dat['TRADECODE'][0][-2:]+sql_dat['TRADECODE'][0][:8]
    today=list(c.execute(sql))[-1][0]
    today=pd.Timestamp(today)
    sql_dat['LAST_DATE']=sql_dat['LAST_DATE'].map(pd.Timestamp)
    sql_dat['days_left']=sql_dat['LAST_DATE']-today
    sql_dat['days_left']=sql_dat['days_left'].map(lambda x: int(x.days))
    sql_selected=sql_dat[np.array(sql_dat['days_left']>=40) & np.array(sql_dat['days_left']<=120)]
    sql_selected=sql_dat[np.array(sql_dat['EXE_PRICE']>=2.3) & np.array(sql_dat['EXE_PRICE']<=2.6)]
    sql_selected=sql_selected[sql_selected['EXE_MODE']=='认购']
    print(len(sql_selected))

def generate_recommend_future(protfolio_id,asset_id,asset_mount,cash):
    pass

# def portfolio_beta():
# cpy实现

# generate_recommend_option_delta('123',['000001.SZ','10001689.SH','10001697.SH'],[100000,0,0],100000)
