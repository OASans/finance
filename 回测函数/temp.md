### generate_recommend_option_delta(protfolio_id：str，asset_id:list<str>,asset_amount:list<int>,cash:float，begin_t:str)

描述：
获取推荐套保的期权名和对应对冲比例值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
后置条件：
返回推荐选择期权id和下一日对冲比例值预测/推荐值：(str,float)

### generate_recommend_option_gamma(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str)

描述：
获取期权套保两期权名及持有比例的预测/推荐值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
后置条件：
返回推荐选择的两期权id和下一日两期权套保持有比例的预测/推荐值：（(str,str),(float,float)）