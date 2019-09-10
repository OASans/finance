## 期权部分函数文档
关于要输入整个投资组合的函数，目前暂定为输入一个id的list<str>和一个表示股数的list<int>以及一个现金额度

以下几个函数计算比较简单，可以交由我们实现或者前端实现……

### portfolio_delta(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str) -> series<float>
### portfolio_gamma(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str) -> series<float>
### portfolio_vega(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str) -> series<float>
### portfolio_theta(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str) -> series<float>
### portfolio_rho(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str) -> series<float>
描述：
获取组合的希腊字母值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数

若取特定某一天的值，则取begin_t=end_t=那天

后置条件：
返回组合的对应时期的一段希腊字母值：series<float>



### portfolio_volatility(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str,time:int) -> series<float>
### portfolio_earning_rate(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str,end_t:str,time:int) -> series<float>
描述：
获取组合的VaR预测、波动率、收益率；计划复用定期调整与条件触发功能部分的实现；
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
time：时间值，即计算1日还是7日还是15日……
后置条件：
返回组合的对应时期的一段值：series<float>

### retrain_delta_model(protfolio_id:str,asset_id:list<str>,asset_amount:list<int>,cash:float,options:str)

描述：

重新训练模型，用于在用户资产组合有调整之后（买卖股票之类的操作之后）或该组合变更了设定方案

protfolio_id为组合的id,需根据该id寻找对于的模型文件

其他同上



### retrain_gamma_model(protfolio_id:str,asset_id:list<str>,asset_amount:list<int>,cash:float,options1:str,options2:str)

描述：

重新训练模型，用于在用户资产组合有调整之后（买卖股票之类的操作之后）或该组合变更了设定方案

protfolio_id为组合的id,需根据该id寻找对于的模型文件

其他同上



### fit_delta(protfolio_id：str,asset_id:list<str>,asset_amount:list<int>,cash:float,options:str,begin_t:str,end_t:str) ->  series<float>

描述：
获取指定期权套保下的推荐对冲比例值
前置条件：

protfolio_id为组合的id,需根据该id寻找对于的模型文件

asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000],期权期货为份数
cash为初始资金数量的浮点数
options为所选期权id
后置条件：
返回时期内对冲比例值预测/推荐值： series<float>

### fit_gamma(protfolio_id：str,asset_id:list<str>,asset_amount:list<int>,cash:float,options1:str,options2:str,begin_t:str,end_t:str) -> series(float,float)
描述：
获取指定两期权套保持有比例的预测/推荐值
前置条件：

protfolio_id为组合的id,需根据该id寻找对于的模型文件

asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
options1、options2为所选期权id
后置条件：
返回时期内期权套保持有比例的预测/推荐值： series(float,float)

### cal_option_amt(total_value:float,option:str,portion:float)->int
描述：
根据组合总价值、期权id、套保比例计算对应所需期权份数
前置条件：
total_value为当下整个组合的总价值
option为期权的id
portion为套保比例，即为对冲比例*完全对冲所需期权比例值
后置条件：
返回期权的份数：int



### generate_recommend_option_delta(protfolio_id：str，asset_id:list<str>,asset_amount:list<int>,cash:float，begin_t:str)->list<str>

描述：
获取推荐套保的期权名和对应对冲比例值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数

begin_t 当前时间

后置条件：
返回推荐选择期权id：list<str>

### generate_recommend_option_gamma(asset_id:list<str>,asset_amount:list<int>,cash:float,begin_t:str)->list<str>

描述：
获取期权套保两期权名及持有比例的预测/推荐值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
后置条件：
返回推荐选择的期权id:list<str>

## 期货部分函数文档

### portfolio_beta(asset_id:list<str>,asset_amount:list<int>,cash:float, begin_t, end_t) -> float

**直接调用陈鹏宇那边实现的beta系数计算**

描述：
获取组合的希腊字母值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
后置条件：
返回组合的对应beta值：float

### retrain_beta_model(protfolio_id:str,asset_id:list<str>,asset_amount:list<int>,cash:float,futures:str)

描述：

重新训练模型，用于在用户资产组合有调整之后（买卖股票之类的操作之后）或该组合变更了设定方案

protfolio_id为组合的id,需根据该id寻找对于的模型文件

其他同上



### fit_beta(protfolio_id：str,asset_id:list<str>,asset_amount:list<int>,cash:float,futures:str,begin_t:str,end_t:str) ->  series<float>
描述：
获取指定期货套保下的推荐对冲比例值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
futures为所选期货id
后置条件：
返回下一日对冲比例值预测/推荐值：float

### cal_future_amt(total_value:float,futures:str,portion:float)->int
描述：
根据组合总价值、期货id、套保比例计算对应所需期货份数
前置条件：
total_value为当下整个组合的总价值
option为期货的id
portion为套保比例，即为对冲比例*完全对冲所需期货比例值
后置条件：
返回期货的份数：int

### generate_recommend_future(asset_id:list<str>,asset_amount:list<int>,cash:float，begin_t:str)->list<str>
描述：
获取推荐套保的期货名和对应对冲比例值
前置条件：
asset_id为id的list ,如['000001.SZ','000002.SZ']
asset_amount为初始资产的配置的list，对于股票单位为股数，如[1000,1000]
cash为初始资金数量的浮点数
后置条件：
返回推荐选择期货id：list<str>