# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:11:26 2019

@author: wbl19
"""


import sqlite3
import pandas as pd
import splitconcept

conn = sqlite3.connect(r'../../获取资产的基本数据/fin_set.db')#连接到db
c = conn.cursor()#创建游标


all_facter=splitconcept.get_all_facter()



def get_stock_info(stock_code: str, column:list) -> pd.DataFrame:
    cnx = sqlite3.connect(r'../../获取资产的基本数据/fin_set.db')#连接到db
    df = pd.read_sql_query("SELECT "+','.join(column)+" FROM "+stock_code+" ORDER BY DATE DESC", cnx)
    return df

def get_market_info(column:list) -> pd.DataFrame:
    cnx = sqlite3.connect(r'../../获取资产的基本数据/fin_set.db')#连接到db
    df = pd.read_sql_query("SELECT "+','.join(column)+" FROM MARKET ORDER BY DATE DESC", cnx)
    return df

def cal_factors():
    
    stockinfo=c.execute('SELECT TRADECODE,CONCEPT,IND_NAME FROM STOCKINFO')
    p=stockinfo.fetchall()
    
    market_df=get_market_info(['DATE','RF','RM'])
    
    for i,each_facter in enumerate(all_facter):
        mv_total=pd.DataFrame(columns=['DATE','PMV','TOTALMV'])
        mv_total.DATE=market_df.DATE
        mv_total.PMV=0
        mv_total.TOTALMV=0
        print(i)
        '''
        这里如果确定date的话就能求某一天的啦
        '''
        for each_stock in p:
            #整理参数
            factor_list=''
            if each_stock[1] is None:
                factor_list=each_stock[2]
            else:
                factor_list=';'.join([each_stock[1],each_stock[2]])
            
            #参数存在
            if (each_facter in factor_list):
                #调整股票名以调表
                stock_code=each_stock[0].split('.')
                stock_code[0],stock_code[1]=stock_code[1],stock_code[0]
                stock_code=''.join(stock_code)
                #取日期、收益率、市值
                try:
                    stock_df=get_stock_info(stock_code,['DATE','PCTCHG','MV'])
                    '''
                    这里如果确定date的话就能求某一天的啦
                    '''
                except:
                    continue
                stock_df.PCTCHG=stock_df.PCTCHG
                
                '''
                暂时数据缺失/
                唉
                '''
                #减无风险收益率
                stock_df.PCTCHG=stock_df.PCTCHG-market_df.RF/360
                #按市值加权
                
                #合并，确保日期一样；已有的值会被新的值替代
                mv_total=pd.merge(mv_total,stock_df,on='DATE',how='outer')
                mv_total.PCTCHG=mv_total.PCTCHG*mv_total.MV
                mv_total.PMV=mv_total[['PMV','PCTCHG']].sum(axis=1, skipna=True,min_count=1)
                mv_total.TOTALMV=mv_total[['TOTALMV','MV']].sum(axis=1, skipna=True,min_count=1)
                mv_total=mv_total.drop(['PCTCHG','MV'],axis=1)
            else:
                continue
        if (mv_total.TOTALMV.all()!=0):
            mv_total.PMV=mv_total.PMV/mv_total.TOTALMV
        #如果这里可以一列添加那就太好了
        #因为我不会，所以改一改方法
        #c.execute('ALTER TABLE MARKET ADD COLUMN F'+str(i+8)+ ' REAL')
        mv_total=mv_total.drop('TOTALMV',axis=1)
        mv_total=mv_total.rename(columns={'PMV':'F'+str(i+8)})
        
        #这里合并取了交集，确保以market为准
        market_df=pd.merge(market_df,mv_total)
    
    print(market_df)
    market_df.to_excel('market_df.xlsx')
    return market_df


