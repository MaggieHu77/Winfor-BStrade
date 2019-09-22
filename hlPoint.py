# -*- coding:utf-8 -*-
# ! python3
import constant
import numpy as np
from copy import copy
from defindex import *
from loadData import *
from misc import printt


class HLPoint:
    def __init__(self, klist, code, thresh=constant.THRESH_D):
        self.klist = klist
        self.code = code
        self.thresh = thresh
        self.hpi = []
        self.lpi = []
        self.confirm_p = []
        self.cursor = 0
        self.hl = None
        self.temp_max = 0
        self.temp_min = 0
        self.temp_h = 0
        self.temp_l = 0
        self.space_h = 0.0
        self.space_l = 0.0
        self.use_space = False
        self.k_lev = 1

    def init_hl(self):
        flag = True  # 是否找到顶分型或底分型
        jj = 0
        while flag:
            if self.klist[jj + 1].high >= self.klist[jj].high and \
                self.klist[jj + 1].high > self.klist[jj + 2].high:
                self.temp_h = jj
                self.hl = "h"
                self.temp_min = jj
                self.klist[jj].temp_h = jj
                self.klist[jj].hl = "h"
                self.klist[jj].temp_min = jj
                self.cursor = jj + 1
                flag = False
                printt(f"##`init_hl`，找到顶分型，位置{jj};"
                       f"当前寻找高点，待判定高点位置{jj},高点价位{self.klist[jj].high}",
                       msg_mode=constant.RunMode.DEBUG,
                       global_mode=constant.RUN_MODE)
            elif self.klist[jj + 1].low <= self.klist[jj].low and \
                self.klist[jj + 1].low < self.klist[jj + 2].low:
                self.temp_l = jj
                self.hl = "l"
                self.temp_max = jj
                self.klist[jj].temp_l = jj
                self.klist[jj].hl = "l"
                self.klist[jj].temp_max = jj
                self.cursor = jj + 1
                flag = False
                printt(f"##`init_hl`，找到底分型，位置{jj};"
                       f"当前寻找低点，待判定低点位置{jj},低点价位{self.klist[jj].low}",
                       msg_mode=constant.RunMode.DEBUG,
                       global_mode=constant.RUN_MODE)
            else:
                jj += 1

    @staticmethod
    def step_hl_s(next_hl, temp_hl_t, high, low, temp_hl, temp_m,
                  from_hl, from_temp, pre_high,
                  pre_low, pre_high2, pre_low2, space_h,
                  space_l, l2h, h2l, stamp, thresh=constant.THRESH_D, **kwargs):

        is_high = False
        is_low = False
        use_space = 0

        if next_hl:
            if high > temp_hl:
                temp_hl = high
                temp_hl_t = stamp
                temp_m = low
                from_hl += from_temp + 1
                from_temp = 0
            else:
                from_temp += 1
                if low < temp_m:
                    temp_m = low
                    if space_h and round((temp_m - temp_hl) / temp_hl, 2) < - space_h * constant.AVG_BUFFER:
                        use_space = 1
                if (use_space or from_temp >= thresh or temp_m < pre_low) and temp_m == low:
                    is_high = True
                    next_hl = 0
                    pre_high2 = pre_high
                    pre_high = temp_hl
                    temp_hl = temp_m
                    temp_hl_t = stamp
                    temp_m = high
                    from_hl = 0
                    from_temp = 0
                    use_space = 0
                    if len(l2h) > constant.AVG_N:
                        space_l = round(np.mean(list(map(lambda x, y: (x - y) / y,
                                                      [k[1] for k in l2h[-constant.AVG_N:]],
                                                      [k[0] for k in l2h[-constant.AVG_N:]]))).item(), 3)
                        if space_l == np.nan:
                            space_l = 0.0
        else:
            if low < temp_hl:
                temp_hl = low
                temp_hl_t = stamp
                temp_m = high
                from_hl += from_temp + 1
                from_temp = 0
            else:
                from_temp += 1
                if high > temp_m:
                    temp_m = high
                    if space_l and (temp_m - temp_hl) / temp_hl > space_l * constant.AVG_BUFFER:
                        use_space = 1
                if (use_space or from_temp >= thresh or temp_m > pre_high) and temp_m == high:
                    is_low = True
                    next_hl = 1
                    pre_low2 = pre_low
                    pre_low = temp_hl
                    temp_hl = temp_m
                    temp_hl_t = stamp
                    temp_m = low
                    from_hl = 0
                    from_temp = 0
                    use_space = 0
                    if len(h2l) > constant.AVG_N:
                        space_h = round(np.mean(list(map(lambda x, y: (x - y)/ x,
                                                      [k[0] for k in h2l],
                                                      [k[1] for k in h2l]))).item(), 3)
                        if space_h == np.nan:
                            space_h = 0.0
        if next_hl and from_hl + from_temp >= constant.WAIT_DTO30:
            lev_chg_signal = True
        else:
            lev_chg_signal = False
        return {"next_hl": next_hl, "temp_hl": temp_hl,
                "temp_m": temp_m, "from_hl": from_hl,
                "from_temp": from_temp, "use_space": use_space,
                "space_h": space_h, "space_l": space_l,
                "is_high": is_high, "is_low": is_low,
                "lev_chg_signal": lev_chg_signal,
                "pre_high": pre_high, "pre_low": pre_low,
                "pre_high2": pre_high2, "pre_low2": pre_low2,
                "temp_hl_t": temp_hl_t}

    def step_hl(self, wait_thresh=constant.WAIT_DTO30):
        """
        完成导入回测期全部K线数据并创建K对象实例list后，在第二次从头遍历K对象实例时的每步迭代函数
        需要完成以下任务：
        / 1.将当前高低点对象实例维护的全局状态域赋值给当前遍历的K对象实例的自有对应域
        :param wait_thresh:
        :return:
        """
        if self.cursor < len(self.klist):
            self.klist[self.cursor].hpi = copy(self.hpi)
            self.klist[self.cursor].lpi = copy(self.lpi)
            self.klist[self.cursor].hl = self.hl
            self.klist[self.cursor].temp_l = self.temp_l
            self.klist[self.cursor].temp_h = self.temp_h
            self.klist[self.cursor].temp_min = self.temp_min
            self.klist[self.cursor].temp_max = self.temp_max
            if self.hl == "h":
                # 在找高点过程中出现新高，则转换当前高点为待判定高点，自新高（当前K）回调最低点为当前K
                if self.klist[self.cursor].high > self.klist[self.temp_h].high:
                    self.temp_h = self.cursor
                    self.temp_min = self.cursor
                    self.klist[self.cursor].temp_h = self.temp_h
                    self.klist[self.cursor].temp_min = self.temp_min
                    printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                           f"寻高点过程出现新高，新待判定高点位置{self.temp_h}，"
                           f"新高点价位{self.klist[self.cursor].high}",
                           msg_mode=constant.RunMode.INFO,
                           global_mode=constant.RUN_MODE)

                else:
                    # 如果找高点过程中未出现新高
                    self.klist[self.cursor].use_space = self.use_space
                    # 如果自待判定高点回调以来出现新低
                    if self.klist[self.cursor].low < self.klist[self.temp_min].low:
                        # 则当前K为回调新低
                        self.temp_min = self.cursor
                        self.klist[self.cursor].temp_min = self.temp_min
                        printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                               f"寻高点过程出现自待判定高点以来回调新低，"
                               f"新低位置{self.cursor},点位{self.klist[self.cursor].low}",
                               msg_mode=constant.RunMode.INFO,
                               global_mode=constant.RUN_MODE)
                        # 判断是否满足以”空间回撤“大小提前（相对于回调时间长度条件）判定高点的条件
                        if self.space_h and \
                                round((self.klist[self.temp_min].low -
                                       self.klist[self.temp_h].high) / self.klist[self.temp_h].high, 2) \
                                < -self.space_h * constant.AVG_BUFFER:
                            self.use_space = True
                            self.klist[self.cursor].use_space = self.use_space
                            printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                                   f"寻高点过程满足空间判定条件，可确认高点",
                                   msg_mode=constant.RunMode.INFO,
                                   global_mode=constant.RUN_MODE)
                    if (self.use_space or (self.temp_min - self.temp_h) >= self.thresh or
                            len(self.lpi) and self.klist[self.temp_min].low < self.klist[self.lpi[-1]].low) and\
                            self.temp_min == self.cursor:
                        # 判定高点的3个条件：
                        # / 0.当前K为自待判定高点以来回调最低点
                        #  以下条件三选一：
                        # / 1.当前K距离带判定高点（时间，指有效日K线数）距离超过阈值-->时间判定
                        # / 2.当前K的最低价距离待判定高点最高价回调幅度超过阈值-->空间判定
                        # / 3.当前K的最低价已经跌破前被确认低点的最低价-->跌破前低判定
                        printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                               f"确认高点，高点位置{self.temp_h},高点价位"
                               f"{self.klist[self.temp_h].high}；当前位置{self.cursor}，转为寻低点"
                               f"过程，待判定低点价位{self.klist[self.cursor].low}",
                               msg_mode=constant.RunMode.DEBUG,
                               global_mode=constant.RUN_MODE)
                        self.hpi.append(self.temp_h)
                        self.klist[self.temp_h].hl_confirmed = self.cursor
                        self.klist[self.cursor].confirm_hl = self.temp_h
                        self.confirm_p.append(self.cursor)
                        # 将全局状态转变为寻找低点相关
                        self.hl = "l"
                        self.klist[self.cursor].hl = 'l'
                        self.temp_l = self.temp_min
                        self.klist[self.cursor].temp_l = self.temp_l
                        self.temp_max = self.cursor
                        self.klist[self.cursor].temp_max = self.temp_max
                        self.use_space = False
                        self.klist[self.cursor].hpi = copy(self.hpi)
                        # 计算从低点到高点的上涨平均空间水平，是为下一次确认低点的条件2做准备
                        l2h = self.l2h()
                        if len(l2h) >= constant.AVG_N:
                            self.space_l = round(np.mean(list(map(lambda x, y: (self.klist[
                                                                                  x].high-self.klist[y].low)/self.klist[y].low,
                                               [k[1] for k in l2h[-constant.AVG_N:]], [k[0] for k in
                                                                                   l2h[
                                                                                   -constant.AVG_N:]]))).item(), 3)
                            if self.space_l == np.nan:
                                self.space_l = 0.0
                            else:
                                printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                                       f"存在低点确认可参考空间涨幅{self.space_l}",
                                       msg_mode=constant.RunMode.INFO,
                                       global_mode=constant.RUN_MODE)

            else:
                # 在找低点过程中
                if self.klist[self.cursor].low < self.klist[self.temp_l].low:
                    # 当前出现跌破待判定低点的前低，更新当前K为待判定低点
                    self.temp_l = self.cursor
                    self.temp_max = self.cursor
                    self.klist[self.cursor].temp_l = self.temp_l
                    self.klist[self.cursor].temp_max = self.temp_max
                    printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                           f"寻低点过程出现新低，新待判定低点位置{self.temp_l}，"
                           f"新低点价位{self.klist[self.cursor].low}",
                           msg_mode=constant.RunMode.INFO,
                           global_mode=constant.RUN_MODE)
                else:
                    self.klist[self.cursor].use_space = self.use_space
                    if self.klist[self.cursor].high > self.klist[self.temp_max].high:
                        # 当前出现自待判定低点上涨以来新高，更新当前K为上涨新高
                        self.temp_max = self.cursor
                        self.klist[self.cursor].temp_max = self.temp_max
                        printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                               f"寻低点过程出现自待判定低点以来上涨新高，"
                               f"新高位置{self.cursor},点位{self.klist[self.cursor].high}",
                               msg_mode=constant.RunMode.INFO,
                               global_mode=constant.RUN_MODE)
                        if self.space_l and \
                                (self.klist[self.temp_max].high -
                                 self.klist[self.temp_l].low) / self.klist[self.temp_l].low > \
                                self.space_l*constant.AVG_BUFFER:
                            self.use_space = True
                            self.klist[self.cursor].use_space = self.use_space
                            printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                                   f"寻低点过程满足空间判定条件，可确认低点",
                                   msg_mode=constant.RunMode.INFO,
                                   global_mode=constant.RUN_MODE)
                    if (self.use_space or (self.temp_max - self.temp_l) >= self.thresh or
                        len(self.hpi) and self.klist[self.temp_max].high > self.klist[self.hpi[-1]].high) and\
                            self.temp_max == self.cursor:
                        # 确认低点的前提条件（条件0）和其他三个任选条件与确认高点是镜像问题
                        printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                               f"确认低点，低点位置{self.temp_l},低点价位"
                               f"{self.klist[self.temp_l].low}；当前位置{self.cursor}，转为寻高点"
                               f"过程，待判定高点价位{self.klist[self.cursor].high}",
                               msg_mode=constant.RunMode.DEBUG,
                               global_mode=constant.RUN_MODE)
                        self.lpi.append(self.temp_l)
                        self.klist[self.temp_l].hl_confirmed = self.cursor
                        self.klist[self.cursor].confirm_hl = self.temp_l
                        self.confirm_p.append(self.cursor)
                        # 将全局状态转换为找高点
                        self.hl = "h"
                        self.klist[self.cursor].hl = 'h'
                        self.temp_h = self.temp_max
                        self.klist[self.cursor].temp_h = self.temp_h
                        self.temp_min = self.cursor
                        self.klist[self.cursor].temp_min = self.temp_min
                        self.use_space = False
                        self.klist[self.cursor].lpi = copy(self.lpi)
                        h2l = self.h2l()
                        if len(h2l) >= constant.AVG_N:
                            self.space_h = round(np.mean(list(map(lambda x, y: (self.klist[
                                                                                  x].high-self.klist[y].low) /
                                                                            self.klist[x].high,
                                               [k[0] for k in h2l[-constant.AVG_N:]], [k[1] for k in h2l[
                                                                                            -constant.AVG_N:]]))).item(), 3)
                            if self.space_h == np.nan:
                                self.space_h = 0.0
                            else:
                                printt(f"##`step_hl-{constant.K_LEV_DICT[self.k_lev]}`，"
                                       f"存在高点确认可参考空间涨幅{self.space_h}",
                                       msg_mode=constant.RunMode.INFO,
                                       global_mode=constant.RUN_MODE)

            if self.hl == "h":
                # 对于找高点的升频条件判定，如果超过等待时间还不能确认高点，发出级别降低提示
                if self.klist[self.cursor].lpi:
                    self.klist[self.cursor].lev_chg_signal = \
                        (self.cursor - self.klist[self.cursor].lpi[-1]) >= wait_thresh
                else:
                    self.klist[self.cursor].lev_chg_signal = self.cursor >= wait_thresh
            self.cursor += 1

    def l2h(self):
        l2h = []
        hpi = self.hpi
        lpi = self.lpi
        if len(hpi) and len(lpi):
            if lpi[0] > hpi[0]:
                hpi = hpi[1:]
            n = min(len(hpi), len(lpi))
            for i in range(n):
                l2h.append((lpi[i], hpi[i]))
        return l2h

    def h2l(self):
        h2l = []
        hpi = self.hpi
        lpi = self.lpi
        if len(hpi) and len(lpi):
            if hpi[0] > lpi[0]:
                lpi = lpi[1:]
            n = min(len(hpi), len(lpi))
            for i in range(n):
                h2l.append((hpi[i], lpi[i]))
        return h2l

    def get_hl(self):
        while self.cursor < len(self.klist):
            self.step_hl()


class HLPointMin(HLPoint):
    def __init__(self, klist_m, code, thresh, k_lev, init_hl):
        '''

        :param klist_m: 分钟K线序列
        :param code: 股票代码
        :param thresh: 高低点参数
        :param k_lev: K线级别
        :param init_hl: 给定初始高低点寻找目标，"h" or "l"
        '''
        super().__init__(klist_m, code, thresh)
        self.superior_stop = False
        self.k_lev = k_lev
        self.hl = init_hl
        self.cursor = 1

    def extend_klist(self, ref_t):
        '''
        扩展本级别分钟级别的数据
        由于在降低级别回测过程中，由日线无法给出分钟级别的结束时间，因此为避免大量提取分钟数据的开销
        采取随着遍历逐渐扩充数据长度的方法，设定初始提取长度，LEN_K30和LEN_K5每当剩余长度不足初始长度1/4时
        提取新数据为原规定长度一半
        :param ref_t: 对应参考的日线时间戳
        :return:
        '''
        len_k = int([constant.LEN_K5 / 2, constant.LEN_K30 / 2][int(self.k_lev == 2)])
        init_i = self.klist[-1].i.kti
        kti_seq = get_kti_seq(list(range(len_k)), init_i, constant.N_30, constant.N_5)
        if self.k_lev == 2:
            begin_time = ref_t[min(kti_seq[1].kti[0], len(ref_t)-1)] + min(constant.MIN30_STR[
                kti_seq[1].kti[1]], "15:00:00")
            end_time = ref_t[min(kti_seq[-1].kti[0], len(ref_t)-1)] + min(constant.MIN30_STR[
                kti_seq[-1].kti[1]], "15:00:00")
        else:
            begin_time = ref_t[min(kti_seq[1].kti[0], len(ref_t)-1)] + min(constant.MIN5_STR[
                kti_seq[1].kti[1]][kti_seq[1].kti[2]], "15:00:00")
            end_time = ref_t[min(kti_seq[-1].kti[0], len(ref_t)-1)] + min(
                           constant.MIN5_STR[kti_seq[-1].kti[1]][kti_seq[-1].kti[2]], "15:00:00")
        self.klist.extend(loadData_min(begin_time,
                     self.code,
                     end_time,
                     [5, 30][int(self.k_lev == 2)],
                     kti_seq[1].kti))


if __name__ == "__main__":
    klist=loadData_daily()
    hlp_env = HLPoint(klist, "600519.SH")
    hlp_env.init_hl()
    while hlp_env.cursor < len(hlp_env.klist):
        hlp_env.step_hl()
    print(f"high points index:{hlp_env.hpi}")
    print(f"low points index:{hlp_env.lpi}")