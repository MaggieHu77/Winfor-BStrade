# -*- coding:utf-8 -*-
# ! python3

from hlPoint import HLPoint
from trend import Trend, TrendError
from pandas import DataFrame, DatetimeIndex, to_datetime, MultiIndex
import matplotlib.pyplot as plt
import numpy as np
from time import strftime, localtime, time
from datetime import date
from dateutil.relativedelta import relativedelta
from matplotlib.dates import AutoDateLocator, DateFormatter
import locale
from loadData import loaddataError, loadData_daily as ldd
from WindPy import w
import constant
from trade import *
from global_backtest import GlobalBacktest
from misc import read_file


class BSgraph(object):
    def __init__(self, trade_env, stat):
        """
        策略绘图对象的创建函数
        :param trade_env: 交易环境操作对象
        :param codename: 股票代码
        """
        self.trade_env = trade_env
        self.stat = stat

    def wrapper_to_global_backtest(self):
        stock_code = np.array([self.stat.trade_env.code])  # 第一索引
        trade_date = [k.t for k in self.stat.trade_env.klist]  # 第二索引
        # 转换为时间类型，只保存date
        trade_date = np.array(to_datetime(trade_date).date)
        wrapper_index = MultiIndex.from_product([stock_code, trade_date], names=["stock_code",
                                                                                "trade_date"])
        wrapper_df = DataFrame(index=wrapper_index)
        wrapper_df.loc[:, "equity"] = self.stat.trade_env.account.equity[1:]
        wrapper_df.loc[:, "cash"] = self.stat.trade_env.account.cash[1:]
        wrapper_df.loc[:, "cumu_r"] = self.stat.trade_env.account.cumu_r[1:]
        wrapper_df.loc[:, "nav"] = wrapper_df.cash + wrapper_df.equity
        wrapper_df.loc[:, "volume"] = self.stat.trade_env.account.volume[1:]
        # 记录买卖点
        wrapper_df.loc[:, "buy_sell"] = "NA"
        wrapper_df.loc[:, "operation_price"] = 0.0
        wrapper_df.loc[:, "operation_volume"] = 0.0
        # 提取买卖点信息
        for row in self.stat.bsp_info.iterrows():
            dd = row[0].date()
            wrapper_df.loc[(stock_code[0], dd), "buy_sell"] = row[1].type
            wrapper_df.loc[(stock_code[0], dd), "operation_price"] = row[1].price
            wrapper_df.loc[(stock_code[0], dd), "operation_volume"] = row[1].volume
        return wrapper_df

    def performance(self,
                    trdchg,
                    dir="",
                    star=False):
        """
        绘图展现回测期内的高低点和趋势变化

        :param codename: 回测股票代码或名称
        :param dir: 图像存放文件夹地址
        :return: 返回图像文件.jpg
        """
        main_df = DataFrame(columns=['date', 'high', 'low', 'close', 'trd'])
        for kk in range(len(self.trade_env.klist)):
            main_df.loc[kk] = [self.trade_env.klist[kk].t, self.trade_env.klist[kk].high,
                               self.trade_env.klist[kk].low, self.trade_env.klist[kk].close,
                               self.trade_env.klist[kk].trd]
        # 以时间为索引
        main_df.index = DatetimeIndex(to_datetime(main_df['date']))
        # 确保升序排列
        main_df.sort_index(ascending=True, inplace=True)

        # 绘图设置部分
        plt.figure(facecolor="white", frameon=True, figsize=(28, 15), dpi=200)
        plt.suptitle(self.trade_env.code + u"择时策略：高低点、趋势及买卖点", size=33)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 提取上升趋势
        up = np.ma.masked_where(
            main_df['trd'].values != 'up', main_df['close'].values
        )
        # 提取下降趋势
        down = np.ma.masked_where(
            main_df['trd'].values != 'down', main_df['close'].values
        )
        # 提取盘整趋势
        consd = np.ma.masked_where(
            main_df['trd'].values != 'consd', main_df['close'].values
        )
        # 提取不能判定趋势部分
        na = np.ma.masked_where(
            main_df['trd'].values, main_df['close'].values
        )
        # Part 1：主图部分
        plt.subplot2grid((3, 9), (0, 0), rowspan=2, colspan=9)
        # 画出各个趋势部分的线
        upline, = plt.plot(
            main_df.index, up, color='red', linestyle='-', label='up'
        )
        downline, = plt.plot(
            main_df.index, down, color='green', linestyle='-', label='down'
        )
        consdline, = plt.plot(
            main_df.index, consd, color='orange', linestyle='-', label='consolidation'
        )
        naline, = plt.plot(
            main_df.index, na, color='grey', linestyle='-', label='unknown'
        )
        # 需要补齐趋势类型之间的空隙
        color_dict = {'up': "red", "down": "green", "consd": "orange", "None": "grey"}
        for ii in trdchg:
            plt.plot(main_df.index[ii-1:ii+1], main_df['close'][ii-1:ii+1],
                     color=color_dict[str(main_df['trd'][ii-1])], linestyle='-')
        # 设置时间标注距离高低点的距离
        scale_text = round(
            (main_df['close'].max() - main_df['close'].min()) / 15, 2
        )
        # 提取高低点信息
        hllist = self.stat.hlp_info
        # 提取日线高点信息
        hlist_d = hllist[(hllist["hl"] == "H") & (hllist["level"] == "daily")]
        # 提取日线低点信息
        llist_d = hllist[(hllist["hl"] == "L") & (hllist["level"] == "daily")]
        # 提取30min高点信息
        hlist_30 = hllist[(hllist["hl"] == "H") & (hllist["level"] == "30min")]
        # 提取30min低点信息
        llist_30 = hllist[(hllist["hl"] == "L") & (hllist["level"] == "30min")]
        # 提取5min高点信息
        hlist_5 = hllist[(hllist["hl"] == "H") & (hllist["level"] == "5min")]
        # 提取5min低点信息
        llist_5 = hllist[(hllist["hl"] == "L") & (hllist["level"] == "5min")]
        # 提取买卖点信息
        bslist = self.stat.bsp_info
        # 提取买点信息
        blist = bslist[bslist["type"].apply(lambda x: "B" in x)]
        # 提取卖点信息
        slist = bslist[bslist["type"].apply(lambda x: "S" in x)]

        # 日线高点信息时间标注
        for hh in range(len(hlist_d)):
            plt.text(
                hlist_d.index[hh],
                hlist_d.ix[hh, 'price'] + scale_text,
                f"H:{hlist_d.ix[hh, 'price']}\n" +
                hlist_d.index[hh].strftime('%m-%d'),
                fontsize=14
            )
        # 日线低点时间标注
        for ll in range(len(llist_d)):
            plt.text(
                llist_d.index[ll],
                llist_d.ix[ll, "price"] - scale_text,
                f"L:{llist_d.ix[ll, 'price']}\n" +
                llist_d.index[ll].strftime('%m-%d'),
                fontsize=14
            )
        # 买点信息时间标注
        ref_klist_t = self.trade_env.klist
        for bb in range(len(blist)):
            x_tick_with_shift = min(blist.ix[bb, 'index_d']+15, len(ref_klist_t)-1)
            plt.text(
                ref_klist_t[x_tick_with_shift].t,
                blist.ix[bb, 'price'] + 2.5*scale_text,
                blist.index[bb].strftime('%m-%d') +
                f'''\n{blist.ix[bb, 'type']}:{blist.ix[bb, 'price']}\n{constant.BUYSELL_TYPE[
                blist.ix[bb, 'type']][3]}''',
                fontsize=14
            )
            plt.annotate("",
                         xy=(blist.index[bb], blist.ix[bb, 'price']),
                         xytext=(ref_klist_t[x_tick_with_shift].t,
                                 blist.ix[bb, 'price'] + 2.5*scale_text),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        # 卖点时间标注
        for ss in range(len(slist)):
            x_tick_with_shift = min(slist.ix[ss, 'index_d'] + 15, len(ref_klist_t)-1)
            plt.text(
                ref_klist_t[x_tick_with_shift].t,
                slist.ix[ss, 'price'] - 3*scale_text,
                slist.index[ss].strftime('%m-%d') +
                f'''\n{slist.ix[ss, 'type']}:{slist.ix[ss, 'price']}\n{constant.BUYSELL_TYPE[
                slist.ix[ss, 'type']][2]}''',
                fontsize=14
            )
            plt.annotate("",
                         xy=(slist.index[ss], slist.ix[ss, 'price']),
                         xytext=(ref_klist_t[x_tick_with_shift].t, slist.ix[ss, 'price'] - 3*scale_text),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        # 高低点记号标注
        # # matplotlib内置的颜色和样式
          # color: b:blue, g:green, r:red, c:cyan, m:magenta, y:yellow, k:black, w:white
          # marker: o:点, D:钻石形，d:细钻石形 ^:上三角， p:五边形， *：五角星 , v:倒三角，<:左三角，>:右三角，s:方形，h/H:六边形
        # 日线高点记号
        high_pd, = plt.plot(
            hlist_d.index, hlist_d["price"], 'r^', markersize=12
        )
        # 日线低点记号
        low_pd, = plt.plot(
            llist_d.index, llist_d["price"], 'gv', markersize=12
        )
        # 30min高点记号标注
        high_p30, = plt.plot(
            hlist_30.index, hlist_30["price"].values, "rx", markersize=10
        )
        # 30min低点记号标注
        low_p30, = plt.plot(
            llist_30.index, llist_30["price"].values, "gx", markersize=10
        )
        # 5min高点记号标注
        high_p5, = plt.plot(
            hlist_5.index, hlist_5["price"].values, "r+", markersize=8
        )
        # 5min低点记号标注
        low_p5, = plt.plot(
            llist_5.index, llist_5["price"].values, "g+", markersize=8
        )
        # 最末日提示收盘价
        if star:
            star_p, = plt.plot(
                main_df.index[-1], main_df.ix[-1, 'close'], 'k*', markersize=13
            )
            plt.text(
                main_df.index[-1], main_df.ix[-1, 'close'] + 0.5*scale_text, '现价:'+str(main_df.ix[
                                                              -1, 'close']), fontsize=14
            )
        # 买卖点记号标注
        # 买点记号
        if not blist.empty:
            buy_p, = plt.plot(
                blist.index, blist["price"], 'r*', markersize=12
            )
        # 卖点记号
        if not slist.empty:
            sell_p, = plt.plot(
                slist.index, slist["price"], 'g*', markersize=12
            )

        # 图例
        if not blist.empty and not slist.empty:
            plt.legend(
                (
                    upline,
                    downline,
                    consdline,
                    naline,
                    high_pd,
                    low_pd,
                    high_p30,
                    low_p30,
                    high_p5,
                    low_p5,
                    buy_p,
                    sell_p
                ),
                (
                    "up trend",
                    "down trend",
                    "consolidation trend",
                    "unknown",
                    "high point",
                    "low point",
                    "high point in 30min",
                    "low point in 30min",
                    "high point in 5min",
                    "low point in 5min",
                    "buy point",
                    "sell point"
                ),
                loc="upper left",
                shadow=False,
                frameon=False,
                fontsize=16,
                facecolor="none",
            )
        elif not blist.empty:
            plt.legend(
                (
                    upline,
                    downline,
                    consdline,
                    naline,
                    high_pd,
                    low_pd,
                    high_p30,
                    low_p30,
                    high_p5,
                    low_p5,
                    buy_p,
                ),
                (
                    "up trend",
                    "down trend",
                    "consolidation trend",
                    "unknown",
                    "high point",
                    "low point",
                    "high point in 30min",
                    "low point in 30min",
                    "high point in 5min",
                    "low point in 5min",
                    "buy point"
                ),
                loc="upper left",
                shadow=False,
                frameon=False,
                fontsize=16,
                facecolor="none",
            )
        else:
            plt.legend(
                (
                    upline,
                    downline,
                    consdline,
                    naline,
                    high_pd,
                    low_pd,
                    high_p30,
                    low_p30,
                    high_p5,
                    low_p5
                ),
                (
                    "up trend",
                    "down trend",
                    "consolidation trend",
                    "unknown",
                    "high point",
                    "low point",
                    "high point in 30min",
                    "low point in 30min",
                    "high point in 5min",
                    "low point in 5min"
                ),
                loc="upper left",
                shadow=False,
                frameon=False,
                fontsize=16,
                facecolor="none",
            )
        # 网格线选项
        plt.grid(True, 'major')
        # x-axis
        plt.xlabel(u"时间", fontsize=18)
        plt.ylabel(u"价格", fontsize=18)
        # 坐标轴时间格式
        ax1 = plt.gca()
        ax1.xaxis.set_major_locator(AutoDateLocator())
        ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)

        # Part 2:第二分图
        plt.subplot2grid((3, 9), (2, 2), rowspan=1, colspan=7)
        # mkt, = plt.plot(DatetimeIndex(self.stat.cumur_mkt.keys()),
        #                 list(self.stat.cumur_mkt.values()),
        #                 color="#808080",
        #                 label="benchmark_cumu_return")
        stk, = plt.plot(DatetimeIndex(self.stat.cumur_stk.keys()),
                        list(self.stat.cumur_stk.values()),
                        color="#808080",
                        label="stock_cumu_return")
        stg_t_seq = main_df.index[main_df.index >= constant.PERFORMANCE_BEGIN_DATE]
        stg, = plt.plot(stg_t_seq, self.trade_env.account.cumu_r[-(len(stg_t_seq)):],
                        color="#DC143C", label="strategy_return")
        plt.legend(
            (stk, stg),
            ("stock", "strategy"),
            loc="upper left",
            frameon=False,
            fontsize=17,
        )
        plt.ylabel(u"累计收益（%）", fontsize=18)
        plt.xlabel(u"时间", fontsize=18)
        ax2 = plt.gca()
        ax2.xaxis.set_major_locator(AutoDateLocator())
        ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

        # Part3: 第三分图，表现统计参数    "
        plt.subplot2grid((3, 9), (2, 0), rowspan=1, colspan=2)
        plt.title(f'''{self.trade_env.code}策略表现对比{constant.BENCHMARK}\nfrom:{
        constant.PERFORMANCE_BEGIN_DATE}''', fontsize=20)
        col_labels = [constant.BENCHMARK, self.trade_env.code, "strategy"]
        row_labels = [u"累计收益", u"年化收益", u"年化标准差", u"收益/标准差", u"最大回撤"]
        cell_text = list(self.stat.performance.values())
        cwid = [0.3, 0.3, 0.3]
        performance_table = plt.table(
            cellText=cell_text,
            rowLoc="center",
            rowLabels=row_labels,
            colLabels=col_labels,
            colLoc="center",
            colWidths=cwid,
            loc="center"
        )
        performance_table.auto_set_font_size(False)
        performance_table.set_fontsize(18)
        table_props = performance_table.properties()
        table_cells = table_props["child_artists"]
        for cell in table_cells:
            cell.set_height(0.14)
        plt.axis("off")
        self.gdir = f"{dir}/BS_{self.trade_env.code}.png"
        # 保存图像
        plt.savefig(self.gdir)
        plt.close()
        return self.gdir

    def strategy_info(self, lasthl, space_h, space_l):
        """
        返回股票在总表中的观测信息
        :param lasthl: 最后一个高点或者低点的index
        :return: Strategy_s表中对应信息tuple
        """
        pass


class Stat:
    def __init__(self, hlp_env, trd_env, trade_env):
        self.hlp_env = hlp_env
        self.trd_env = trd_env
        self.trade_env = trade_env
        self.hlp_info = None
        self.bsp_info = None
        self.mkt_p = {}
        self.cumur_stk = {}
        self.cumur_mkt = {}
        self.performance = {}

    def get_hl_info(self):
        hlp = []
        # 记入日线高点
        for kdh in self.trade_env.hpi:
            hlp.append((self.trade_env.klist[kdh].t,
                        self.trade_env.klist[kdh].high,
                        "H",
                        constant.K_LEV_DICT[1],
                        self.trade_env.klist[self.trade_env.klist[kdh].hl_confirmed].t))
        # 记入日线低点
        for kdl in self.trade_env.lpi:
            hlp.append((self.trade_env.klist[kdl].t,
                        self.trade_env.klist[kdl].low,
                        "L",
                        constant.K_LEV_DICT[1],
                        self.trade_env.klist[self.trade_env.klist[kdl].hl_confirmed].t))
        # 记入30min高点
        for k30h in self.trade_env.hp30:
            hlp.append((k30h.t, k30h.high, "H", constant.K_LEV_DICT[2], ""))
        for k30l in self.trade_env.lp30:
            hlp.append((k30l.t, k30l.low, "L", constant.K_LEV_DICT[2], ""))
        # 记入5min高点
        for k5h in self.trade_env.hp5:
            hlp.append((k5h.t, k5h.high, "H", constant.K_LEV_DICT[3], ""))
        for k5l in self.trade_env.lp5:
            hlp.append((k5l.t, k5l.low, "L", constant.K_LEV_DICT[3], ""))
        hlp.sort()
        self.hlp_info = DataFrame(hlp, columns=["date", "price", "hl", "level", "confirmed_date"])
        self.hlp_info.index = DatetimeIndex(self.hlp_info["date"])
        self.hlp_info.sort_index(ascending=True, inplace=True)
        return hlp

    def get_bs_info(self):
        bsp = []
        for bs in self.trade_env.bs_list:
            if "Trade.Buy" in str(type(bs)):
                bsp.append((bs.t,
                            bs.index_k.kti[0],
                            self.trade_env.bs_list.index(bs),
                            bs.price,
                            bs.volume,
                            bs.typ,
                            bs.stop_p, bs.max_profit, bs.cumu_r, bs.operate_lev,
                            self.trade_env.bs_list.index(bs.sell_p) if bs.sell_p else -1))
            else:
                bsp.append((bs.t, bs.index_k.kti[0], self.trade_env.bs_list.index(bs), bs.price,
                            bs.volume, bs.typ,
                            0.0, 0.0, 0.0, 1, self.trade_env.bs_list.index(bs.buy_p)))
        bsp.sort()
        self.bsp_info = DataFrame(bsp, columns=["date", "index_d", "index", "price", "volume",
                                                "type", "stop_p", "max_r",
                                                "cumu_r", "lev_chg", "counter_opt"])
        self.bsp_info.index = DatetimeIndex(self.bsp_info["date"])
        self.bsp_info.sort_index(ascending=True, inplace=True)
        return bsp

    def get_account_info(self):
        account = self.trade_env.account
        return (self.trade_env.klist[-1].t, self.trade_env.code, account.cash[-1],
                account.volume[-1], account.equity[-1], account.cumu_r[-1])

    def get_strategy_info(self):
        pass

    def get_performance(self):
        assert len(self.trade_env.account.cumu_r) - 1 == \
               len(self.trade_env.klist), \
            f"{self.trade_env.code}: account cumulative returns length {len(self.trade_env.account.cumu_r)} " \
            f"doesn't match K objects length {len(self.trade_env.klist)}"
        performance_t_seq = np.array([k.t for k in self.trade_env.klist])
        performance_begin_index = int(np.where(performance_t_seq >=
                                               constant.PERFORMANCE_BEGIN_DATE)[0][0])
        stg_mkt = list(np.array(self.trade_env.account.cash[(performance_begin_index + 1):]) +
                         np.array(self.trade_env.account.equity[(performance_begin_index + 1):]))
        stk_mkt = [k.close for k in self.trade_env.klist][performance_begin_index:]
        r_stg = list(np.diff(stg_mkt) / np.array(stg_mkt[:-1]))
        mkt_p = list(self.mkt_p.values())
        # 累计收益
        self.cumur_mkt = dict(zip(self.mkt_p.keys(),
                                  list(map(lambda x, y: x/y - 1, mkt_p, [mkt_p[0]] * len(mkt_p)))))
        # 股票自身累计收益
        self.cumur_stk = dict(zip(list(performance_t_seq[performance_begin_index:]),
                                  list(map(lambda x, y: x/y-1, stk_mkt, [stk_mkt[0]]*len(
                                      stk_mkt)))))
        cumu_r_mkt = format(mkt_p[-1] / mkt_p[0] - 1, ".2%")
        cumu_r_stg = format(self.trade_env.account.cumu_r[-1], ".2%")
        cumu_r_stk = format(self.trade_env.klist[-1].close / self.trade_env.klist[
            performance_begin_index].close - 1, ".2%")
        # 年化收益
        stk_p = [k.close for k in self.trade_env.klist[performance_begin_index:]]
        r_stk = list(np.diff(stk_p) / np.array(stk_p[:-1]))
        r_mkt = list(np.diff(mkt_p) / np.array(mkt_p[:-1]))

        annu_r_mkt = format(np.nanmean(r_mkt).item() * constant.N_TRADE, ".2%")
        annu_sd_mkt = format(np.nanstd(r_mkt).item() * constant.N_TRADE ** 0.5, ".2%")
        r_sd_ratio_mkt = round(np.nanmean(r_mkt).item() * constant.N_TRADE ** 0.5/
                               np.nanstd(r_mkt).item(), 2)
        annu_r_stk = format(np.nanmean(r_stk).item() * constant.N_TRADE, ".2%")
        annu_sd_stk = format(np.nanstd(r_stk).item() * constant.N_TRADE ** 0.5, ".2%")
        r_sd_ratio_stk = round(np.nanmean(r_stk).item() * constant.N_TRADE ** 0.5/
                               np.nanstd(r_stk).item(), 2)
        annu_r_stg = format(np.nanmean(r_stg).item() * constant.N_TRADE, ".2%")
        annu_sd_stg = format(np.nanstd(r_stg).item() * constant.N_TRADE**0.5, ".2%")
        try:
            r_sd_ratio_stg = round(np.nanmean(r_stg).item() * constant.N_TRADE ** 0.5 /
                                   np.nanstd(r_stg).item(), 2)
        except ZeroDivisionError:
            r_sd_ratio_stg = "/"
        max_drawdown_stk = self.max_drawdown(stk_p)
        max_drawdown_stg = self.max_drawdown(stg_mkt)
        max_drawdown_mkt = self.max_drawdown(mkt_p)
        self.performance = {u"累计收益": [cumu_r_mkt, cumu_r_stk, cumu_r_stg],
                u"年化收益": [annu_r_mkt, annu_r_stk, annu_r_stg],
                u"年化标准差": [annu_sd_mkt, annu_sd_stk, annu_sd_stg],
                u"收益/标准误": [r_sd_ratio_mkt, r_sd_ratio_stk, r_sd_ratio_stg],
                u"最大回撤": [max_drawdown_mkt, max_drawdown_stk, max_drawdown_stg]}

    def max_drawdown(self, series):
        # 回撤开始时间
        jj = np.argmax(np.maximum.accumulate(series) - series)
        # 回撤结束时间
        try:
            ii = np.argmax(series[:jj])
        except ValueError:
            ii = 0
        return format(float(series[jj]) / float(series[ii]) - 1.0, ".2%")

    def get_mkt_p(self):
        if not w.isconnected():
            w.start()
        res = w.wsd(constant.BENCHMARK, 'close', max(self.trade_env.klist[0].t,
                                                     constant.PERFORMANCE_BEGIN_DATE), self.trade_env.klist[-1].t)
        res_t = [t.strftime("%Y-%m-%d 15:00:00") for t in res.Times]
        mkt_p = res.Data[0]
        self.mkt_p.update(dict(zip(res_t, mkt_p)))


# 运行参数设置函数
def runbacktest(
        begin,
        codename,
        dir,
        end,
        paint=True,
        star=False
):

    """
    设置策略环境参数，并运行部分回测策略
    :param begin: 回测开始时间
    :param codename: 回测股票代码或名称
    :param dir: 作图目录地址
    :param end: 回测结束时间，默认当前
    :param paint: bool值，是否画图，默认为True
    :param star: bool值，最后一个标记星星，默认为False
    :return: 返回策略环境对象整体和图像地址
    """
    print(f"BS日线策略：正在回测{codename}...")
    try:
        klist = ldd(begin, codename, end)
    except loaddataError as e:
        print(f"\t{e}")
        raise loaddataError
    print(f"\t{codename}获取日K线数{len(klist)}")
    # 由于klist数据集是被hlp_env, trd_env和trade_env共享的，指向同一地址
    # 因此任意指针均可调用
    # 设置策略运行参数环境
    hlp_env = HLPoint(klist, codename)
    hlp_env.init_hl()
    hlp_env.get_hl()
    try:
        trd_env = Trend(hlp_env)
        trd_env.get_trend()
    except TrendError as e:
        print(e)
        return
    try:
        trade_env = Trade(code=codename,
                          klist=hlp_env.klist,
                          hpi=hlp_env.hpi,
                          lpi=hlp_env.lpi)
        trade_env.get_bs()
    except Exception as e:
        print(e)
        return
    for k in trade_env.klist:
        k.t += " 15:00:00"  # 统一日线时间与分钟级别时间的表示方法 "yyyy-mm-dd HH:MM:SS"
    # note = [[1, hlp_env.temp_h, hlp_env.temp_min],
    #         [0, hlp_env.temp_l, hlp_env.temp_max]][hlp_env.hl == "l"]
    locale.setlocale(locale.LC_CTYPE, "chinese")
    # 创建回测表现统计操作对象
    stat = Stat(hlp_env=hlp_env, trd_env=trd_env, trade_env=trade_env)
    stat.get_mkt_p()
    hlp_tb = stat.get_hl_info()
    bsp_tb = stat.get_bs_info()
    account_row = stat.get_account_info()
    strategy_row = stat.get_strategy_info()
    # 打包回测结果总结
    # res_pac = {"hlp": hlp_tb, "bsp": bsp_tb, "account": account_row, "strategy": strategy_row}
    stat.get_performance()
    # 创建图像操作对象
    graph_obj = BSgraph(trade_env=trade_env, stat=stat)
    # 向回测全局对象返回结果总结
    wrapper_df = graph_obj.wrapper_to_global_backtest()
    # 该股票在总表中的信息
    if paint:
        # 绘图并获取图像地址
        gdir = graph_obj.performance(
            trdchg=trd_env.trdchg,
            dir=dir,
            star=star,
        )
        print(f"回测图像输出到{gdir}")
        # return res_pac, gdir
        return wrapper_df
    else:
        # return res_pac
        return wrapper_df


if __name__ == "__main__":
    constant.reset_params()
    constant.check_dir()
    start_time = constant.BEGIN_DATE
    end_time = constant.END_DATE
    # 读入对象
    codes = read_file(constant.CODE_FILE)
    # 创建全局回测对象
    # global_backtest_obj = GlobalBacktest(start_time=start_time,
    #                                      end_time=end_time)
    # global_backtest_obj.run_backtest(global_index="000300.SH",
    #                                  func=runbacktest,
    #                                  key_params=dict(begin=start_time, end=end_time,
    #                                                  dir=constant.GRAPH))
    for code in codes:
        runbacktest(begin=start_time, end=end_time,
                    dir=constant.GRAPH, codename=code,
                    )








