import pandas as pd
import sqlite3
from bigreader import *
from WindPy import *
import tushare as ts

pro = ts.pro_api()
# w.start()

def init():
    conn = sqlite3.connect('fin_set.db')
    c = conn.cursor()
    return conn, c


def is_tradedate(today):
    alldays = ts.trade_cal()
    tradedays = alldays[alldays['isOpen'] == 1]  # 开盘日
    # today = datetime.today().strftime('%Y-%m-%d')
    if today in tradedays['calendarDate'].values:
        return True
    else: return False

def update_stock(conn, c , today):
    cursor = conn.execute("SELECT tradecode from STOCKINFO")
    i = 1
    for row in cursor:
        if i > 100:
            break
        i = i + 1
        stk = row[0]
        df1 = pro.daily(ts_code=stk, trade_date=today.replace('-', ''))
        df2 = pro.daily_basic(
            ts_code=stk, trade_date=today.replace('-', ''), fields=
            'ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,total_share'
        )
        df = pd.merge(df1, df2)
        df['trade_date'] = df['trade_date'].apply(lambda d: transdate2(str(d)))
        fetch = w.wsd(
            stk,
            "swing,tech_turnoverrate20,tech_turnoverrate60,tech_turnoverrate240,west_netprofit_YOY,profit_ttm,west_netprofit_FY3,fa_egro,fa_sgro",
            today,
            today,
            "Fill=Previous;PriceAdj=F",
            usedf=True)
        df3 = fetch[1].reset_index()
        df3 = df3.rename(columns={"index": "ts_code"})
        df = pd.merge(df, df3)
        df = df.drop(columns="ts_code")
        col_names = [
            'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'PRECLO', 'CHG', 'PCTCHG',
            'VOLUME', 'AMT', 'TURN', 'VR', 'PE', 'PB', 'TOTALSHARE', 'MV',
            'SWING', 'STOM', 'STOQ', 'STOA', 'EST_NETPGROWTH', 'NETPROFIT',
            'EST_NETP3', 'EGRO', 'SGRO'
        ]
        df.columns = col_names
        syb = transname(stk)
        cur = conn.execute("SELECT * from " + syb)
        row = [len(cur.fetchall())] + list(df.iloc[0, :])
        c.execute('INSERT INTO ' + syb + ' VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', row)
        conn.commit()
        # pd.io.sql.to_sql(df, syb, con=conn, if_exists='append')


def update_go():
    today = datetime.today().strftime('%Y-%m-%d')
    conn, c = init()
    if is_tradedate(today):
        update_stock(conn, c, today)
    else: print("Today is not a trading day.")
