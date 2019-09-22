# -*- coding: utf-8 -*-
# ! python3
import constant
from misc import printt
from loadData import loadData_min
from math import floor
from hlPoint import HLPointMin
from defindex import *
import numpy as np
from defindex import Kti


class Trade:
    def __init__(self, code, klist, hpi, lpi):
        self.code = code
        self.klist = klist
        self.operate_lev = 1
        self.operate_lev_buy = 1
        self.operate_lev_sell = 1
        self.hpi = hpi
        self.lpi = lpi
        self.bs_list = []  # 所有发生过的买卖记录，按照时间先后排序
        self.order_l = []  # 当前未平仓的买入交易
        self.order_s = []  # 发生过的卖出交易
        self.cursor = 0
        self.account = Trade.Account(trade_len=len(klist))
        self.hp30 = []
        self.lp30 = []
        self.hp5 = []
        self.lp5 = []

    class Buy:
        """
        定义买点对象相关性质和方法
        """
        def __init__(self, time: str, index: Kti, typ: int, price: float, account,
                     stop_p: float = None):
            self.t = time
            self.typ = typ
            self.index_k = index
            self.price = price
            self.volume = 0
            self.account = account
            self.stop_p = stop_p
            self.max_profit = 0.0
            self.cumu_r = 0.0
            self.operate_lev = 1  # 针对此买点的寻找卖点操作级别
            self.closed = False
            self.sell_p = None
            self.lev_chg = None  # 级别转变操作对象，用于寻找特殊卖点

        def register(self):
            self.volume = self.account.buy(typ=self.typ, price=self.price,
                                           buy_idx=self.index_k.kti[0])

        def sold(self, sell_p):
            """
            卖出操作记录：对应卖点对象实例，清仓标记
            :param sell_p: 卖点对象实例
            :return:
            """
            self.sell_p = sell_p
            self.closed = True

        def refresh(self, current_p: float, current_high: float, cursor: int):
            """
            每个K线出现后，更新买点收益状态
            :param current_p: 当前价格
            :param current_high: 当前最高价
            :param cursor: 刷新价格当日索引
            """
            buy_idx = self.index_k.kti[0]
            if cursor >= buy_idx:
                self.cumu_r = round(current_p / self.price - 1, 3)  # 更新累计收益
                cumu_with_r = round(current_high / self.price - 1, 3)  # 更新用最高价计算的累计收益
                if self.max_profit < constant.THRESH_PROFIT and cumu_with_r >= constant.THRESH_PROFIT:
                    # 在第一次累计收益达到阈值的时候，更新止盈价格（抬高）基准
                    self.reset_stop_p()
                self.max_profit = max(self.max_profit, self.cumu_r)

        def reset_stop_p(self):
            self.stop_p = self.price * (1 + constant.THRESH_STOPPROFIT)  # 止盈点参考价格设为买入价加上一定保底收益

    class Sell:
        """
        定义卖点对象相关性质和方法
        """
        def __init__(self, time, index, typ, volume, price, account, buy_p):
            self.t = time
            self.typ = typ
            self.index_k = index
            self.price = price
            self.volume = volume
            self.account = account
            self.buy_p = buy_p  # 卖点对象知道其对应买点对象是谁，但买点对象暂时不知道其卖点是谁

        def register(self):
            """
            卖点操作发生：”通知”账户对象实例记录本次交易，并处理其历史对应买点
            :return:
            """
            self.account.sell(price=self.price, volume=self.volume, sell_idx=self.index_k.kti[0])
            self.buy_p.closed = True
            self.buy_p.sell_p = self

    class Account:
        """
        记录交易及账户资产变动情况
        """
        def __init__(self, trade_len, volume=0.0, equity=0.0, cumu_r=0.0):
            self.trade_len = trade_len  # 回测期的长度，注意多出一格作为初始状态
            self.cash = np.repeat(constant.INIT_FUND, self.trade_len+1).astype(np.float32)  # 账户现金
            self.buy2_pct = 1 / (1 - constant.BUY1_PCT) * constant.BUY2_PCT
            self.volume = np.repeat(volume, self.trade_len+1).astype(np.float32)  # 持有手数-->累计值
            self.equity = np.repeat(equity, self.trade_len+1).astype(np.float32)  # 股权价值
            self.cumu_r = np.repeat(cumu_r, self.trade_len+1).astype(np.float32)  # 累计收益

        def buy(self, typ: str, price: float, buy_idx: int):
            """
            定义买入操作
            :param typ: 买点级别
            :param price: 买入价位
            :param buy_idx: 买入操作发生时对应的日K索引
            :return: delta_vol，买入手数
            """
            info = constant.BUYSELL_TYPE[typ]
            buy_idx += 1
            if info[1] == 1:
                # 第一类买点
                delta_vol = floor(self.cash[buy_idx] * constant.BUY1_PCT / price)  # 计算交易量
                self.volume[buy_idx:] = self.volume[buy_idx] + delta_vol  # 当前持仓总手数
                self.cash[buy_idx:] = self.cash[buy_idx] - delta_vol * price  # 当前账户剩余现金
                self.equity[buy_idx:] = self.volume[buy_idx] * price  # 按当前买入价格计算，资产账户总市值
                self.cumu_r[buy_idx:] = round((self.cash[buy_idx] +
                                               self.equity[buy_idx]) / self.cash[0] - 1,
                                              4)  # 账户现金+资产市值的累计收益率
            else:
                # 第二、三类买点
                delta_vol = round(self.cash[buy_idx] * self.buy2_pct / price)
                self.volume[buy_idx:] = self.volume[buy_idx] + delta_vol
                self.cash[buy_idx:] = self.cash[buy_idx] - delta_vol * price
                self.equity[buy_idx:] = self.volume[buy_idx] * price
                self.cumu_r[buy_idx:] = round((self.cash[buy_idx] +
                                               self.equity[buy_idx]) / self.cash[0] - 1, 4)
            return delta_vol

        def sell(self, price, volume, sell_idx):
            """
            定义卖出操作
            :param price: 卖出价格
            :param volume: 卖出手数
            :param sell_idx: 卖出时日K的序号索引

            """
            sell_idx += 1
            self.volume[sell_idx:] = self.volume[sell_idx] - volume  # 剩余持仓手数
            self.cash[sell_idx:] = self.cash[sell_idx] + price * volume  # 当前账户现金总数
            self.equity[sell_idx:] = self.volume[sell_idx] * price  # 剩余资产以卖出价格计价市值
            self.cumu_r[sell_idx:] = round((self.cash[sell_idx] +
                                            self.equity[sell_idx]) / self.cash[0] - 1,
                                           4)  # 当前账户总值计价累计收益率

        def sum_up(self, current_p: float, cursor: int):
            """
            当日K结束后更新账户状态
            :param current_p: 当前价格，一般指收盘价
            :param cursor: 当前日K索引

            """
            cursor += 1
            self.equity[cursor:] = self.volume[cursor] * current_p
            self.cumu_r[cursor:] = \
                round((self.cash[cursor] + self.equity[cursor]) / self.cash[0] - 1, 4)

    @staticmethod
    def seek_to_buy_s(pre_low, pre_high30, num_h30, num_l30,
                      pre_low30, stamp, trend, pre_trend, index,
                      account, high, low, price, last_bs_typ):
        """
        用于单步买点查找，必须有30min的高低点至少三个，高点在前；才能有效构成买入
        :param pre_low: 点日线低点
        :param pre_high30: 前30min高点
        :param pre_low30: 前30min低点
        :param stamp: 当前时间戳
        :param trd: 当前趋势
        :param index: 如果确认买点的编号
        :param typ: 如果确认买点的类型
        :param account: Account对象
        :param high: 当前最高价
        :param low: 当前最低价
        :param price: 如果满足条件，以此价格买入
        :param last_bs_typ: 上一个买卖操作的类型
        :return:
        """
        flag = False
        operate_lev_buy = 2
        buy_p = None
        if trend == 'down' and low > pre_low:
            res = Trade.buy_in_min30_s(low=low, high=high, pre_low=pre_low, pre_high30=pre_high30,
                                       stamp=stamp, index=index, typ="B1-0", price=price,
                                       account=account, pre_low30=pre_low30, num_h30=num_h30,
                                       num_l30=num_l30)
            if res[0]:
                operate_lev_buy = 1
            elif res[1]:
                operate_lev_buy = 1
                flag = True
                buy_p = res[2]
        elif trend in ("consd", "up") and pre_trend == "down" and low > pre_low:
            res = Trade.buy_in_min30_s(low=low, high=high, pre_low=pre_low, pre_high30=pre_high30,
                                       stamp=stamp, index=index, typ="B2-0", price=price,
                                       account=account, pre_low30=pre_low30, num_h30=num_h30,
                                       num_l30=num_l30)
            if res[0]:
                operate_lev_buy = 1
            elif res[1]:
                operate_lev_buy = 1
                flag = True
                buy_p = res[2]
        elif trend == "consd" and pre_trend == "up" and low > pre_low:
            res = Trade.buy_in_min30_s(low=low, high=high, pre_low=pre_low, pre_high30=pre_high30,
                                       stamp=stamp, index=index, typ="B2-0", price=price,
                                       account=account, pre_low30=pre_low30, num_h30=num_h30,
                                       num_l30=num_l30)
            if res[0]:
                operate_lev_buy = 1
            elif res[1]:
                operate_lev_buy = 1
                flag = True
                buy_p = res[2]
        elif trend == "up" and low > pre_low and last_bs_typ in ("S4", "S5"):
            res = Trade.buy_in_min30_s(low=low, high=high, pre_low=pre_low, pre_high30=pre_high30,
                                       stamp=stamp, index=index, typ="B3-0", price=price,
                                       account=account, pre_low30=pre_low30, num_h30=num_h30,
                                       num_l30=num_l30)
            if res[0]:
                operate_lev_buy = 1
            elif res[1]:
                operate_lev_buy = 1
                flag = True
                buy_p = res[2]
        return flag, operate_lev_buy, buy_p

    def seek_to_buy(self):
        """
        处理买入操作
        :return: 是否发生买入操作
        """
        flag = False  # 是否发生买入操作
        h_confirmed = [self.klist[hh].hl_confirmed for hh in self.hpi]  # 回测期间内所有高点被确认时的K线序号
        if (self.cursor in h_confirmed) and len(self.klist[self.cursor].lpi):  #
            # 如果本K线当日确认了高点，且非第一个高低点
            last_opt = "NA" if not self.bs_list else self.bs_list[-1].typ
            pre_low = self.klist[self.cursor].lpi[-1]  # 前低点序列号
            pre_low_i = self.lpi.index(pre_low)  # 前低点在低点序列内的序列号
            # 如果前低点不是回测区间内最后一个低点，那么30min数据提取范围从本K线到后一低点被确认当日结束；否则从本K线到回测期最后一日
            # 此处不涉及未来函数，因为在逐日监控中，30min买点寻找在低点被确认当日自然停止
            l_confirmed = len(self.klist) - 1 if pre_low == self.lpi[-1] else \
                self.klist[self.lpi[pre_low_i+1]].hl_confirmed
            # 左侧买点，下跌过程中，高点确认时尚未跌破前低（存在下一低点比前低点抬高的可能）
            if self.klist[self.cursor].trd == 'down' and \
                self.klist[self.cursor].low > self.klist[pre_low].low:
                self.operate_lev_buy = 2  # 转换买入操作级别为30min级别
                printt(f"##`seek_to_buy`，尝试寻找第一买点，位置{self.cursor}；"
                       f"确认高点处于下跌趋势中，目前未跌破前低点；"
                       f"30min搜索范围{self.cursor}~{l_confirmed}",
                       msg_mode=constant.RunMode.DEBUG,
                       global_mode=constant.RUN_MODE)
                res = self.buy_in_min30(self.cursor, l_confirmed, pre_low, "B1-0")
                if res[0]:
                    self.operate_lev_buy = 1

                elif res[1]:
                    self.operate_lev_buy = 1
                    flag = True
                else:
                    if pre_low_i < len(self.lpi)-1:
                        printt(f"##`seek_to_buy`，转而尝试寻找日线第一买点，位置{self.cursor}",
                               msg_mode=constant.RunMode.DEBUG,
                               global_mode=constant.RUN_MODE)
                        # NOTE: 不需要额外增加判定日线低点抬高的条件，因为如果中间跌破前低，res[0]为True
                        flag = self.buy_in_day(self.lpi[pre_low_i+1], "B1-1")
                        # self.operate_lev_buy = [2, 1][flag]
                        self.operate_lev_buy = 1
            elif self.klist[self.cursor].trd == "consd" and \
                    self.klist[self.cursor].pre_trd == "down" and \
                    self.klist[self.cursor].low > self.klist[pre_low].low:
                # 由下跌转为盘整趋势中（说明被确认的高点高于前高，而前两个低点降低），目前尚未跌破前低
                self.operate_lev_buy = 2
                printt(f"##`seek_to_buy`，尝试寻找第二买点，位置{self.cursor}；"
                       f"确认高点处于下跌趋势转为盘整，目前未跌破前低点；"
                       f"30min搜索范围{self.cursor}~{l_confirmed}",
                       msg_mode=constant.RunMode.DEBUG,
                       global_mode=constant.RUN_MODE)
                res = self.buy_in_min30(self.cursor, l_confirmed, pre_low, "B2-0")
                if res[0]:
                    self.operate_lev_buy = 1

                elif res[1]:
                    self.operate_lev_buy = 1
                    flag = True
                else:
                    if pre_low_i < len(self.lpi)-1:
                        flag = self.buy_in_day(self.lpi[pre_low_i+1], "B2-1")
                        # self.operate_lev_buy = [2, 1][flag]
                        self.operate_lev_buy = 1
            elif self.klist[self.cursor].trd == "up" and \
                    self.klist[self.cursor].low > self.klist[pre_low].low and last_opt[:2] not in \
                    ("B2", "B3"):
                # 当前处于上涨趋势中，且上一操作不是第二买点或第三买点
                self.operate_lev_buy = 2
                printt(f"##`seek_to_buy`，尝试寻找第二（三）买点，位置{self.cursor}；"
                       f"确认高点处于上涨趋势中，目前未跌破前低点；"
                       f"30min搜索范围{self.cursor}~{l_confirmed}",
                       msg_mode=constant.RunMode.DEBUG,
                       global_mode=constant.RUN_MODE)
                res = self.buy_in_min30(self.cursor, l_confirmed, pre_low, ["B2-0", "B3-0"][int(
                    last_opt in ("S4", "S5"))])
                if res[0]:
                    self.operate_lev_buy = 1

                elif res[1]:
                    self.operate_lev_buy = 1
                    flag = True
                else:
                    if pre_low_i < len(self.lpi)-1:
                        flag = self.buy_in_day(self.lpi[pre_low_i+1], ["B2-1", "B3-1"][int(
                    last_opt in ("S4", "S5"))])
                        # self.operate_lev_buy = [2, 1][flag]
                        self.operate_lev_buy = 1
            elif self.klist[self.cursor].trd == "consd" and \
                self.klist[self.cursor].pre_trd == "up" and \
                self.klist[self.cursor].low > self.klist[pre_low].low and \
                "B1" not in last_opt:
                # 由上涨转入盘整趋势中，高点低点区间连续放宽
                self.operate_lev_buy = 2
                printt(f"##`seek_to_buy`，尝试寻找第买点一，位置{self.cursor}；"
                       f"上涨趋势转为盘整，目前未跌破前低点；"
                       f"30min搜索范围{self.cursor}~{l_confirmed}",
                       msg_mode=constant.RunMode.DEBUG,
                       global_mode=constant.RUN_MODE)
                res = self.buy_in_min30(self.cursor, l_confirmed, pre_low, "B1-0")
                if res[0]:
                    self.operate_lev_buy = 1

                elif res[1]:
                    self.operate_lev_buy = 1
                    flag = True
                else:
                    if pre_low_i < len(self.lpi)-1:
                        flag = self.buy_in_day(self.lpi[pre_low_i+1], "B1-1")
                        # self.operate_lev_buy = [2, 1][flag]
                        self.operate_lev_buy = 1
        return flag

    @staticmethod
    def buy_in_min30_s(low, high, pre_low, pre_high30,
                       stamp, index, typ, price,
                       account, pre_low30, num_h30, num_l30):
        flag1 = False
        flag2 = False
        buy_p = None
        if low < pre_low:
            flag1 = True
        else:
            if num_h30 + num_l30 >= 3:
                if high > pre_high30:
                    flag2 = True
                    buy_p = Trade.Buy(stamp, index, typ, price, account, pre_low30)
                    buy_p.register()
        return flag1, flag2, buy_p

    def buy_in_min30(self, start, end, pre_low, typ):
        flag1 = False  # 是否跌破前日线低点，跌破日线前低点，终止买点搜索
        flag2 = False  # 是否找到买点机会
        begin_time = self.klist[start + 1].t + constant.MIN30_STR[0]
        end_time = self.klist[end].t + constant.MIN30_STR[-1]
        klist30 = loadData_min(begin_time, self.code, end_time, 30, start + 1)
        hlp30_env = HLPointMin(klist30, self.code, constant.THRESH_30, 2, "l")
        for cursor in range(hlp30_env.cursor, len(hlp30_env.klist)):
            ref_low30 = -np.inf  # 如果要以30min低点作参考， 暂时不需要
            # # 如果已经有30min级别低点，不能跌破前低
            # if len(hlp30_env.klist[cursor].lpi):
            #     ref_low30 = hlp30_env.klist[hlp30_env.klist[cursor].lpi[-1]].low

            if hlp30_env.klist[cursor].low < max(self.klist[pre_low].low, ref_low30):
                flag1 = True  # 买点寻找过程中是否跌破前日线低点或更保守的以30min前低点作为参照
                printt(f"###`buy_in_min30`，未发现30min买点；"
                       f"失败理由：30min下跌趋势中跌破前日线(或前30min)低点",
                       msg_mode=constant.RunMode.REPORT,
                       global_mode=constant.RUN_MODE)
                break
            else:
                hlp30_env.step_hl(wait_thresh=constant.WAIT_30TO5) # step_hl使得cursor增加1，注意减回来
                if len(hlp30_env.hpi) + len(hlp30_env.lpi) < 3:
                    # printt(f"###`buy_in_min30`，未发现30min买点；"
                    #        f"失败理由：30min未构成完整趋势",
                    #        msg_mode=constant.RunMode.REPORT,
                    #        global_mode=constant.RUN_MODE)
                    pass
                else:
                    if len(hlp30_env.klist[cursor].hpi) >= 1:  # NOTE:放宽要求，l1-h1-l2-cursor(
                        # c>h2, l2>l1)模式也承认
                        if hlp30_env.klist[cursor].high > \
                                hlp30_env.klist[hlp30_env.klist[cursor].hpi[-1]].high and \
                                (hlp30_env.klist[hlp30_env.klist[cursor].lpi[-1]].low >
                                 hlp30_env.klist[hlp30_env.klist[cursor].lpi[-2]].low):
                            flag2 = True  # 是否找到符合条件的30min级别买点
                            # 记录有效（有交易发生）30min高点K对象
                            self.hp30 += \
                                [hlp30_env.klist[cur] for cur in hlp30_env.hpi]
                            # 记录有效（有交易发生）30min低点K对象
                            self.lp30 += \
                                [hlp30_env.klist[cur] for cur in hlp30_env.lpi]
                            stop_p = hlp30_env.klist[hlp30_env.klist[cursor].temp_l].low
                            stop_p *= (1+constant.STOP_P_BUFFER_30)
                            buy_p = Trade.Buy(hlp30_env.klist[cursor].t,
                                              hlp30_env.klist[cursor].i,
                                              typ,
                                              hlp30_env.klist[cursor].close,
                                              self.account,
                                              stop_p=stop_p
                                              )
                            printt(f"###`buy_in_min30`，发现30min买点，买点类型:{typ}；"
                                   f"买入理由：30min下跌趋势中突破前30min高点；"
                                   f"止损价格{stop_p}->30min最近低点",
                                   msg_mode=constant.RunMode.REPORT,
                                   global_mode=constant.RUN_MODE)
                            buy_p.register()
                            self.order_l.append((buy_p, len(self.bs_list)))
                            self.bs_list.append(buy_p)
                            self.klist[hlp30_env.klist[cursor].i.kti[0]].trade_info(len(
                                self.bs_list) - 1,
                                                                                    typ,
                                                                                    hlp30_env.klist[cursor].close,
                                                                                    buy_p.volume,
                                                                                    buy_p.stop_p
                                                                                    )
                            break
        return flag1, flag2

    @staticmethod
    def buy_in_day_s(pre_low, typ, high, stamp, index, price, account):
        flag = False
        buy_p = None
        if high <= pre_low * (1 + constant.MAX_UPFLOAT):
            buy_p = Trade.Buy(stamp, index, typ, price, account, pre_low)
            flag = True
            buy_p.register()
        return flag, buy_p

    def buy_in_day(self, l_p, typ):
        """
        在日线级别上的买入操作需要满足以下条件
        / 1.30min级别买入失败
        / 2.日线低点被确认当日K收盘
        / 3.确认当日K的最高价距离被确认的低点上涨幅度低于阈值
        :param l_p: 最近被确认日线低点索引
        :param typ: 卖点类型
        :return: 是否发生日线买入操作
        """
        flag = False  # 是否发生买入操作
        l_pp = self.klist[l_p].low
        l_confirmed = self.klist[l_p].hl_confirmed
        p = self.klist[l_confirmed].high
        if p <= l_pp * (1 + constant.MAX_UPFLOAT):
            buy_p = Trade.Buy(self.klist[l_confirmed].t,
                              self.klist[l_confirmed].i,
                              typ,
                              self.klist[l_confirmed].close,
                              self.account,
                              l_pp)
            printt(f"##`buy_in_day`，发现日线级别买点，位置{l_confirmed}，买点类型{typ}；"
                   f"买入价格{self.klist[l_confirmed].close}；"
                   f"买入理由：日线级别低点被确认时上涨幅度未超过阈值；"
                   f"止损价格{l_pp}-->被确认日线低点",
                   msg_mode=constant.RunMode.REPORT,
                   global_mode=constant.RUN_MODE)
            flag = True
            buy_p.register()
            self.order_l.append((buy_p, len(self.bs_list)))
            self.bs_list.append(buy_p)
            self.klist[l_confirmed].trade_info(len(self.bs_list) - 1,
                                               buy_p.typ,
                                               buy_p.price,
                                               buy_p.volume,
                                               buy_p.stop_p)
        else:
            printt(f"##`buy_in_day`，未发现日线级别买点；"
                   f"失败理由：日线级别低点被确认时上涨幅度超过阈值",
                   msg_mode=constant.RunMode.REPORT,
                   global_mode=constant.RUN_MODE)
        return flag

    def seek_to_sell(self):
        '''
        此函数用于寻找卖点
        :return:
        '''
        flag = False  # 是否发生卖出操作
        # 如果没有多头仓位，不操作
        if not self.order_l:
            pass
        else:
            # 先以日线卖出标准将能够卖出的都卖出
            for b in self.order_l:
                bb = b[0]  # b =(buy object, index in order_l)
                # 符合日线上止损点或止盈点卖出条件， 注意不能在同一天交易
                if self.klist[self.cursor].low < bb.stop_p and self.klist[self.cursor].i.kti[0] > \
                        bb.index_k.kti[0]:
                    # 如果是浮盈后离场
                    if bb.max_profit > constant.THRESH_PROFIT:
                        typ = "S3"
                    # 其他止损点离场
                    else:
                        typ = "S2"
                    # 创建卖点对象
                    sell_p = Trade.Sell(self.klist[self.cursor].t,
                                        self.klist[self.cursor].i,
                                        typ,
                                        bb.volume,
                                        self.klist[self.cursor].close,
                                        self.account,
                                        bb)
                    printt(f"##`seek_to_sell`，发现卖点，位置{self.cursor}；"
                           f"卖点类型{typ}，卖出价格{self.klist[self.cursor].close}；"
                           f"卖出理由：{constant.BUYSELL_TYPE[typ][2]}",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                    flag = True
                    # 在Account中注册卖点信息
                    sell_p.register()
                    self.bs_list.append(sell_p)
                    self.order_l.remove(b)
                    self.order_s.append((sell_p, len(self.bs_list)-1))
                    self.klist[self.cursor].trade_info(len(self.bs_list)-1,
                                                       typ,
                                                       sell_p.price,
                                                       sell_p.volume,
                                                       None
                                                       )
                elif self.klist[self.cursor].low < self.klist[self.klist[self.cursor].lpi[
                    -1]].low and self.klist[self.cursor].i.kti[0] > bb.index_k.kti[0]:
                    # 不符合上述条件，但是直接跌破前日线低点；一般适用于在日线级别买入后马上跌破
                    # 或30min低点恰好是日线低点马上跌破的30min买点
                    typ = "S1"
                    sell_p = Trade.Sell(self.klist[self.cursor].t,
                                        self.klist[self.cursor].i,
                                        typ,
                                        bb.volume,
                                        self.klist[self.cursor].close,
                                        self.account,
                                        bb)
                    printt(f"##`seek_to_sell`，发现卖点，位置{self.cursor}；"
                           f"卖点类型S1，卖出价格{self.klist[self.cursor].close}；"
                           f"卖出理由：{constant.BUYSELL_TYPE['S1'][2]}",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                    flag = True
                    sell_p.register()
                    self.bs_list.append(sell_p)
                    self.order_l.remove(b)
                    self.order_s.append((sell_p, len(self.bs_list)-1))
                    self.klist[self.cursor].trade_info(len(self.bs_list) - 1,
                                                       typ,
                                                       sell_p.price,
                                                       sell_p.volume,
                                                       None
                                                       )
            # 当日不能以日线卖出的买点，考虑是否需要进行降级别处理
            # 降级别计算较复杂，主要功能采用一个特别类对象来处理
            if len(self.order_l):
                if self.klist[self.cursor].trd == "up" and \
                        self.klist[self.cursor].lev_chg_signal:
                    for b in self.order_l:
                        bb = b[0]
                        # 当前时间买点已经发生
                        if self.klist[self.cursor].i.kti > bb.index_k.kti:
                            if not bb.lev_chg:
                                bb.lev_chg = Trade.LevChg(code=self.code,
                                                          lev=2,
                                                          start=self.klist[self.cursor + 1],
                                                          ref_t=[k.t for k in self.klist]
                                                          )
                                printt(f"##`sek_to_sell`，发起30min级别卖点寻找，"
                                       f"开始位置{self.cursor}",
                                       msg_mode=constant.RunMode.REPORT,
                                       global_mode=constant.RUN_MODE)
                            else:
                                bb.lev_chg.roll_forward(self.klist[self.cursor])
                                if bb.lev_chg.end_up_trd:
                                    sell_k = bb.lev_chg.sell_k
                                    typ = ["S4", "S5"][sell_k.lev == 3]
                                    self.hp30 += \
                                        [bb.lev_chg.klist[cur] for cur in bb.lev_chg.hlp_env.hpi]
                                    self.lp30 += \
                                        [bb.lev_chg.klist[cur] for cur in bb.lev_chg.hlp_env.lpi]
                                    if typ == "S5":
                                        self.hp5 += bb.lev_chg.hp5
                                        self.lp5 += bb.lev_chg.lp5
                                    sell_p = Trade.Sell(sell_k.t,
                                                        sell_k.i,
                                                        typ,
                                                        bb.volume,
                                                        sell_k.close,
                                                        self.account,
                                                        bb)
                                    printt(f"##`seek_to_sell`，发现卖点，位置{self.cursor}；"
                                           f"卖点类型{typ}，卖出价格{sell_k.close}；"
                                           f"卖出理由：{constant.BUYSELL_TYPE['S1'][2]}",
                                           msg_mode=constant.RunMode.REPORT,
                                           global_mode=constant.RUN_MODE)
                                    flag = True
                                    sell_p.register()
                                    self.bs_list.append(sell_p)
                                    self.order_l.remove(b)
                                    self.order_s.append((sell_p, len(self.bs_list) - 1))
                                    self.klist[self.cursor].trade_info(len(self.bs_list) - 1,
                                                                       typ,
                                                                       sell_p.price,
                                                                       sell_p.volume,
                                                                       None
                                                                       )
                        else:
                            continue
        return flag

    def get_bs(self):
        """
        寻找买卖点主函数
        :return:
        """
        while self.cursor < len(self.klist):
            # 当日是否满足买入条件，发生买入交易
            buy_flag = self.seek_to_buy()
            # 对现有买点更新其每笔多头持仓信息
            for bb in self.order_l:
                bb[0].refresh(self.klist[self.cursor].close, self.klist[self.cursor].high,
                              self.cursor)
            # 当日是否满足卖出条件，发生卖出交易
            sell_flag = self.seek_to_sell()
            # 总账户信息刷新
            self.account.sum_up(self.klist[self.cursor].close,
                                self.cursor)
            # 在经过卖点判断后，如果还存在未平仓的多头头寸
            for bb in self.order_l:
                # 确定每笔多头的操作级别
                if bb[0].lev_chg:
                    # 如果监控该买点卖出操作，存在级别转换对象（处于变小操作频率状态），至少操作级别为2（30min及以上）
                    bb[0].operate_lev = 2
                    if bb[0].lev_chg.lev_chg_lower:
                        # 如果该买点的级别转换对象还存在更小级别的频率对象，则当前处于5min操作级别
                        bb[0].operate_lev = 3
            # 当前卖点操作级别为所有未平仓买点操作级别最高者
            self.operate_lev_sell = max([bb[0].operate_lev for bb in self.order_l] + [1])
            # 当前买卖操作级别为买点操作级别和卖点操作级别孰高
            self.operate_lev = max(self.operate_lev_buy, self.operate_lev_sell)
            self.cursor += 1

    class LevChg:
        def __init__(self, code, lev, start, ref_t):
            self.code = code  # 股票代码
            self.lev = lev  # 本级别，2=30 min or 3=5 min
            self.lev_chg_lower = None  # 是否有低级别对象，None or LevChg object(expect to be lev=3)
            self.klist = []
            self.start = start
            self.ref_t = ref_t  # 上一级别的的K对象，为确定进入本级别的后一个上级别单位
            self.hlp_env = None  # 高低点操作对象
            self.end_up_trd = False
            self.sell_k = None  # 可能出现的卖点K线
            self.init_klist()
            self.hp5 = []
            self.lp5 = []

        def init_klist(self):
            if self.lev == 2:
                t1, t2, t3 = self.start.i.kti
                begin_time = self.start.t + constant.MIN30_STR[0]
                kti_seq = get_kti_seq(list(range(1, constant.LEN_K30+1)), (t1 - 1, t2, t3), constant.N_30, constant.N_5)
                end_time = self.ref_t[min(kti_seq[-1].kti[0], len(self.ref_t)-1)] + min(
                           constant.MIN30_STR[kti_seq[
                    -1].kti[1]], "15:00:00")
                self.klist = loadData_min(begin_time, self.code, end_time, 30, t1)
                self.hlp_env = HLPointMin(klist_m=self.klist,
                                          code=self.code,
                                          thresh=constant.THRESH_30,
                                          k_lev=2,
                                          init_hl="h")  # 默认寻找高点

            else:
                t1, t2, t3 = self.start.i.kti
                begin_time = self.ref_t[min(self.start.i.kti[0], len(self.ref_t)-1)] + \
                             min(constant.MIN5_STR[t2][0], "15:00:00")
                kti_seq = get_kti_seq(list(range(1, constant.LEN_K5+1)), (t1, t2-1, t3), constant.N_30, constant.N_5)
                end_time = self.ref_t[min(kti_seq[-1].kti[0], len(self.ref_t)-1)] + min(
                           constant.MIN5_STR[
                    kti_seq[
                    -1].kti[1]][
                    kti_seq[-1].kti[2]], "15:00:00")
                self.klist = loadData_min(begin_time, self.code, end_time, 5, (t1, t2))
                self.hlp_env = HLPointMin(klist_m=self.klist,
                                          code=self.code,
                                          thresh=constant.THRESH_5,
                                          k_lev=3,
                                          init_hl="h")

        def roll_forward(self, lev_higher):
            while self.hlp_env.cursor < len(self.klist) and \
                    self.klist[self.hlp_env.cursor].i <= lev_higher.i:
                self.hlp_env.step_hl(wait_thresh=constant.WAIT_30TO5)  # step_hl操作会使cursor向后推
                # 如果有同时在进行的低级别过程
                if self.lev_chg_lower:
                    self.lev_chg_lower.roll_forward(self.klist[self.hlp_env.cursor-1])
                    if self.lev_chg_lower.end_up_trd:
                        self.end_up_trd = True
                        printt(f"###`roll_forward`，30min->5min级别下上涨趋势结束，"
                               f"位置{lev_higher.i.kti}",
                               msg_mode=constant.RunMode.REPORT,
                               global_mode=constant.RUN_MODE)
                        self.sell_k = self.lev_chg_lower.sell_k
                        self.hp5 = [self.lev_chg_lower.hlp_env.klist[cur] for
                                    cur in self.lev_chg_lower.hlp_env.hpi]
                        self.lp5 = [self.lev_chg_lower.hlp_env.klist[cur] for
                                    cur in self.lev_chg_lower.hlp_env.lpi]
                        break
                # 没有更低级别的进程
                else:
                    # 是否满足本级别终止条件
                    # 如果当前级别至少有一个低点
                    # 由于默认应当已经在上涨趋势，一旦跌破前低就认为是结束本级别上涨趋势
                    if self.hlp_env.lpi:
                        if self.klist[self.hlp_env.cursor-1].low < \
                                self.klist[self.hlp_env.lpi[-1]].low:
                            self.end_up_trd = True
                            self.sell_k = self.klist[self.hlp_env.cursor-1]
                            printt(f"###`roll_forward`，"
                                   f"{['30min', '5min'][self.lev-2]}级别下上涨趋势结束，"
                                   f"位置{lev_higher.i.kti}，发出卖点信号",
                                   msg_mode=constant.RunMode.REPORT,
                                   global_mode=constant.RUN_MODE)
                            break
                    # 不满足本级别终止条件，是否需要进入低接别
                    if self.lev == 2 and self.klist[self.hlp_env.cursor-1].lev_chg_signal:
                        self.lev_chg_lower = Trade.LevChg(code=self.code,
                                                          lev=3,
                                                          start=self.klist[self.hlp_env.cursor],
                                                          ref_t=self.ref_t)
                        printt(f"###`roll_forward`，"
                               f"由30min操作级别转入5min操作级别寻找卖点，"
                               f"位置{lev_higher.i.kti}",
                               msg_mode=constant.RunMode.REPORT,
                               global_mode=constant.RUN_MODE)
            # 完成本轮循环，说明本级别过程仍然继续，需要考虑扩充本级别序列的数据
            if len(self.klist) - self.hlp_env.cursor <= [constant.LEN_K5*0.25,
                                                         constant.LEN_K30*0.25][self.lev == 2]:
                self.hlp_env.extend_klist(self.ref_t)


if __name__ == "__main__":
    pass









