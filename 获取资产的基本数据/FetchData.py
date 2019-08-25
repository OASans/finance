from WindPy import *
import time,datetime

class FetchData:
    def now(self):
        return time.strftime('[%Y-%m-%d %H:%M:%S]', time.localtime(time.time()))  # 获取当前时间

    def fetchInit(self):
        w.start() # 初始化WindPy接口

    def fetchStockList(self,type):
        if type == 'AStock': #A股
            sectorId = 'a001010100000000'
        if type == 'CFuture': #商品期货
            sectorId = '1000015512000000'
        if type == 'FFuture': #金融期货
            sectorId = 'a599010101000000'
        info = 'date='+str(date.today())+';sectorId='+sectorId+';field=wind_code'  # 组合查询语句
        syblist = w.wset("sectorconstituent", info, usedf=False)    # 发起查询
        sybs = syblist.Data[0]  # 获得数据list
        return sybs # 返回所选板块成分列表

    def fetchRealtimeSnapshot(self,sybs):
        # for syb in sybs:
        #     # output_file = './' + syb + '.xlsx'
        #     got = w.wsq(syb, "rt_time,rt_last,rt_last_amt,rt_last_vol,rt_latest")
        #     # got = w.wsq(syb, "rt_time,rt_last,rt_last_amt,rt_last_vol,rt_latest", func=DemoWSQCallback)
        #     # output = got[1]
        #     # output.to_excel(output_file)
        got = w.wsq(sybs, "rt_time,rt_last,rt_last_amt,rt_last_vol,rt_latest")  # 发起查询
        return got  # 双休日不开盘，当前没有实时数据，无法测试，测试后再尝试修改成写入数据库的形式
    # 估计要使用订阅的方法，在单机上实时向数据库推送新数据

if __name__ == '__main__':
    run = FetchData
    run.fetchInit()
    wantedType = 'AStock'
    sybs = run.fetchStockList(wantedType)
    output = run.fetchRealtimeSnapshot(sybs)
    output.to_sql('...')  # 待补充



