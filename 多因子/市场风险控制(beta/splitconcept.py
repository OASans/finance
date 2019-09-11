# -*- coding: utf-8 -*-
"""
author:wbl19

For course

"""

import sqlite3


def get_all_facter():
    conn = sqlite3.connect(r'../../获取资产的基本数据/fin_set.db')#连接到db
    c = conn.cursor()#创建游标
    stockinfo=c.execute('SELECT TRADECODE,CONCEPT,IND_NAME FROM STOCKINFO')
    p=stockinfo.fetchall()
    for i,each in enumerate(p):
        if (None in each):
            p[i]=[]
        else:
            p[i]=each[1].split(';')
    
    #这样p就是每个股票的概念
            
    all_concepts=[]
    for each in p:
        all_concepts.extend(each)        
    all_concepts=list(set(all_concepts))
    
    #all_concepts是概念表
    
    stockinfo=c.execute('SELECT IND_NAME FROM STOCKINFO')
    p=stockinfo.fetchall()
    all_ind=[]
    for each in p:
        all_ind.extend(list(each))
    all_ind=list(set(all_ind))

    #all_ind是行业表

    all_facter=all_ind+all_concepts
    return all_facter