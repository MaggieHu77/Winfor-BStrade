# -*- coding:utf-8 -*-
# ! python3

from WindPy import w
from datetime import date as dd
from K import K, K30, K5
import constant
from defindex import Kti
from numpy import nan
from defindex import get_kti_seq


class loaddataError(Exception):
    def __init__(self, msg):
        self.errorinfo = msg

    def __str__(self):
        return self.errorinfo


def loadData_daily(begin_date=constant.BEGIN_DATE, stockname='600519.SH',
                   end_date=constant.END_DATE):
    if not w.isconnected():
        w.start()

    res = w.wsd(stockname, "high, low, close, trade_status", begin_date, end_date,
                'priceadj=F', showblank=0)
    is_index = w.wss(stockname, 'windtype').Data[0][0] == "股票指数"
    K_list = []
    if res.ErrorCode != 0:
        #print(stockname + " load daily K info Error: wsd - " +
         #     str(res.ErrorCode))
        # 这里抛出定义的异常，能够在调动的上层捕捉，以防程序异常停止
        raise loaddataError(stockname + 'load data from Wind error: ' +
                            res.ErrorCode)
    # TODO:优化对非停牌日导致的价格数据缺失的前向填充方法，借用pd.DataFrame的方法
    valid_idx = 0
    for jj in range(len(res.Data[0])):
        if not is_index and res.Data[3][jj] == "停牌一天":
            continue
        if jj >= 1:
            res.Data[0][jj] = (res.Data[0][jj] or res.Data[0][jj - 1])
            res.Data[1][jj] = (res.Data[1][jj] or res.Data[1][jj - 1])
            res.Data[2][jj] = (res.Data[2][jj] or res.Data[2][jj - 1])
        if not res.Data[0][jj] or not \
                res.Data[1][jj] or not res.Data[2][jj]:
            continue
        temp_time = res.Times[jj].strftime("%Y-%m-%d")
        # DEBUG: Kti标记需要剔除掉停牌期
        k = K(time=temp_time, high=round(res.Data[0][jj], 2),
              low=round(res.Data[1][jj], 2), close=round(res.Data[2][jj], 2),
              i=Kti(8, valid_idx, 7, 5), lev=1)
        K_list.append(k)
        valid_idx += 1
    return K_list


def loadData_min(begin_time, stockname, end_time, barsize, init_p):
    if not w.isconnected():
        w.start()
    res = w.wsi(stockname, "high, low, close", begin_time, end_time,
                f"barsize={barsize}; Priceadj=F")
    K_list = []
    if res.ErrorCode != 0:
        #print(f"Error:{stockname} load {barsize}min K info error: wsi-{res.ErrorCode}")
        raise loaddataError(f"{stockname} load min data from Wind error: {res.ErrorCode}")
    else:
        if barsize == 30:
            if not isinstance(init_p, tuple):
                init_p = (init_p, 0, constant.N_5 - 1)
            seq_kti = get_kti_seq(list(range(len(res.Data[0]))), init_p, constant.N_30, constant.N_5)
            n = -1
            for jj in range(len(res.Data[0])):

                if res.Data[0][jj] == nan or res.Data[1][jj] == nan or res.Data[2][jj] == nan:
                    continue
                n += 1
                temp_t = res.Times[jj].strftime("%Y-%m-%d %H:%M:%S")
                k = K30(high=round(res.Data[0][jj], 2),
                        low=round(res.Data[1][jj], 2),
                        close=round(res.Data[2][jj], 2),
                        i=seq_kti[n],
                        lev=2,
                        time=temp_t
                        )
                K_list.append(k)
        else:
            init_p = (init_p[0], init_p[1], 0)
            seq_kti = get_kti_seq(list(range(len(res.Data[0]))), init_p, constant.N_30, 1)
            n = -1
            for jj in range(len(res.Data[0])):
                if res.Data[0][jj] == nan or res.Data[1][jj] == nan or res.Data[2][jj] == nan:
                    continue
                n += 1
                temp_t = res.Times[jj].strftime("%Y-%m-%d %H:%M:%S")
                k = K5(high=round(res.Data[0][jj], 2),
                       low=round(res.Data[1][jj], 2),
                       close=round(res.Data[2][jj], 2),
                       i=seq_kti[n],
                       lev=3,
                       time=temp_t)
                K_list.append(k)
        return K_list







