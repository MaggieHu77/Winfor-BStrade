# -*- coding:utf-8 -*-
# ! python3

import constant
from misc import printt


class TrendError(Exception):
    def __init__(self, msg="Undecidable trend; Need more high or low points to decide trend"):
        self.errorinfo = msg

    def __str__(self):
        return self.errorinfo


class Trend:
    """
    趋势判断操作对象，注意趋势判断不涉及分钟数据操作
    """
    def __init__(self, hlp_env):
        self.hlp_env = hlp_env
        self.cursor = 0
        self.trdchg = []
        self.trdnow = None

    def init_trd(self):
        if len(self.hlp_env.hpi) + len(self.hlp_env.lpi) < 4:
            raise TrendError(f"TrendError:{self.hlp_env.code}高低点个数不足，无法继续策略已被忽略")
        else:
            if self.hlp_env.hpi[0] > self.hlp_env.lpi[0]:  # 低点先出现
                if self.hlp_env.klist[self.hlp_env.lpi[1]].low > self.hlp_env.klist[self.hlp_env.lpi[0]].low:
                    # 低点升高，推断为上升趋势，趋势开始时间为第二个低点被确认
                    self.cursor = self.hlp_env.klist[self.hlp_env.lpi[1]].hl_confirmed
                    self.hlp_env.klist[self.cursor].trd = 'up'
                    self.trdchg.append(self.cursor)
                    self.trdnow = 'up'
                    printt(f"##`init_trd`，从l1-h1-l2(l2>l1)推断为上涨趋势",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                else:
                    self.cursor = self.hlp_env.klist[self.hlp_env.lpi[1]].hl_confirmed
                    self.hlp_env.klist[self.cursor].trd = 'consd'
                    self.trdchg.append(self.cursor)
                    self.trdnow = 'consd'
                    printt(f"##`init_trd`，从l1-h1-l2(l2<l1)推断为盘整趋势",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)

            else:  # 高点先出现
                if self.hlp_env.klist[self.hlp_env.hpi[1]].high < self.hlp_env.klist[self.hlp_env.hpi[0]].high:  # 高点降低
                    self.cursor = self.hlp_env.klist[self.hlp_env.hpi[1]].hl_confirmed
                    self.hlp_env.klist[self.cursor].trd = 'down'
                    self.trdchg.append(self.cursor)
                    self.trdnow = 'down'
                    printt(f"##`init_trd`，从h1-l1-h2(h2<h1)推断为下跌趋势",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                else:  # 高点抬高
                    self.cursor = self.hlp_env.klist[self.hlp_env.hpi[1]].hl_confirmed
                    self.hlp_env.klist[self.cursor].trd = 'consd'  # 高-低-高（抬高）-->盘整
                    self.trdchg.append(self.cursor)  # 记录趋势改变（或首次出现能被判定的趋势类型时）
                    self.trdnow = 'consd'
                    printt(f"##`init_trd`，从h1-l1-h2(h2>h1)推断为盘整趋势",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
            self.cursor += 1

    @staticmethod
    def step_trdmax_s(hl, low, high, from_temp, from_hl, temp_hl, trd):
        flag = False
        if not hl and trd != 'down':
            if from_temp + from_hl >= constant.TREND_REV:
                if low <= temp_hl:
                    trd = 'down'
                    flag = True
        elif hl and trd != 'up':
            if from_temp + from_hl >= constant.TREND_REV:
                if high >= temp_hl:
                    trd = 'up'
                    flag = True
        return trd, flag

    def step_trdmax(self):
        """
        趋势转换判定的时间超跌超涨法
        :return:
        """
        flag = False
        if self.hlp_env.klist[self.cursor].hl == 'l' and self.trdnow != 'down':  # 当前待判定的是低点，当前趋势不是下跌
            pre_h = self.hlp_env.klist[self.cursor].hpi[-1]
            if self.cursor - pre_h >= constant.TREND_REV:  # 自前高超过一定时间还未出现低点
                interval = [ii.low for ii in self.hlp_env.klist[pre_h: (self.cursor + 1)]]
                if self.cursor == interval.index(min(interval)) + pre_h:
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'down'
                    self.trdchg.append(self.cursor)
                    self.hlp_env.klist[self.cursor].trd = 'down'
                    flag = True
                    self.cursor += 1
                    printt(f"##`step_trdmax`，确认下跌趋势开启，位置{self.cursor-1}；"
                           f"之前趋势:{self.hlp_env.klist[self.cursor-1].pre_trd}；"
                           f"确认条件：超过{constant.TREND_REV}未出现低点",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
        elif self.hlp_env.klist[self.cursor].hl == 'h' and self.trdnow != 'up': # 待判定高点，当前趋势非上涨
            pre_l = self.hlp_env.klist[self.cursor].lpi[-1]
            if self.cursor - pre_l >= constant.TREND_REV:
                interval = [ii.high for ii in self.hlp_env.klist[pre_l: (self.cursor + 1)]]
                if self.cursor == interval.index(max(interval)) + pre_l:
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'up'
                    self.trdchg.append(self.cursor)
                    self.hlp_env.klist[self.cursor].trd = 'up'
                    flag = True
                    self.cursor += 1
                    printt(f"##`step_trdmax`，确认上涨趋势开启，位置{self.cursor-1}；"
                           f"之前趋势:{self.hlp_env.klist[self.cursor-1].pre_trd}；"
                           f"确认条件：超过{constant.TREND_REV}未出现高点",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
        return flag

    @staticmethod
    def step_trd_s(trd, hl, low, high, pre_low, pre_high, pre2_low, pre2_high):
        if trd == "up" and hl:
            if low < pre_low:
                trd = "consd"
        elif trd =="up" and not hl:
            if low < pre_low:
                if pre_high < pre2_high:
                    trd = 'down'
                else:
                    trd = "consd"
        elif trd == 'down' and not hl:
            if high > pre_high:
                trd = 'consd'
        elif trd == 'down' and hl:
            if high > pre_high:
                if pre_low > pre2_low:
                    trd = 'up'
                else:
                    trd = 'consd'
        elif trd == 'consd' and hl:
            if pre_low > pre2_low:
                if high > pre_high:
                    trd = "up"
        elif trd == 'consd' and not hl:
            if pre_high < pre2_high:
                if low < pre_low:
                    trd = 'down'
        return trd

    def step_trd(self):
        """
        趋势判定主函数，在高低点标定之后进行
        :return:
        """
        if self.trdnow == 'up' and self.hlp_env.klist[self.cursor].hl == 'h':
            # 当前上涨趋势+待判定高点
            if self.hlp_env.klist[self.cursor].low < self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-1]].low:
                # NOTE：该种情况可能不会发生，因为一旦跌破前低点，满足确认高点条件，当即确认高点，hl转变为”l“
                self.trdchg.append(self.cursor)  # 上升趋势中，待判定高点未确认，若期间跌破前低，转为盘整
                self.hlp_env.klist[self.cursor].trd = 'consd'
                self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                self.trdnow = 'consd'
                printt(f"##`step_trd`，确认盘整趋势开启，位置{self.cursor}；"
                       f"之前趋势：up；确认条件：上涨趋势中高点未确认，跌破前低点",
                       msg_mode=constant.RunMode.REPORT,
                       global_mode=constant.RUN_MODE)
            else:
                self.hlp_env.klist[self.cursor].trd = self.trdnow  # 延续上升趋势
                self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[self.cursor - 1].pre_trd
        elif self.trdnow == 'up' and self.hlp_env.klist[self.cursor].hl == 'l':  # 当前上涨趋势+待判定低点
            if self.hlp_env.klist[self.cursor].low < self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-1]].low:
                # 当前跌破前低点，上涨趋势终止
                if self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-1]].high < \
                    self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-2]].high:
                    # 高点已连续降低，可追认下跌趋势
                    self.hlp_env.klist[self.cursor].trd = 'down' # 上升趋势，待判定低点，期间跌破前低，且高点已连续降低
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    # 上升趋势转为下跌-->跌破前低表明上升结束，追认高点降低为下跌条件
                    self.trdchg.append(self.cursor)
                    self.trdnow = 'down'
                    printt(f"##`step_trd`，确认下跌趋势开启，位置{self.cursor}；"
                           f"之前趋势：up；确认条件：上涨趋势中待判定低点，期间跌破前低点，高点已确认并连续降低，追认下跌趋势",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                else:
                    self.hlp_env.klist[self.cursor].trd = 'consd'   # 上升趋势，待判定低点，期间跌破前低，表明上升结束
                    self.trdchg.append(self.cursor)  # 但高点未连续降低，尚不满足下跌条件，转为盘整
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'consd'
                    printt(f"##`step_trd`，确认盘整趋势开启，位置{self.cursor}；"
                           f"之前趋势：up；确认条件：上涨趋势中待判定低点，期间跌破前低点，高点已确认但未连续降低",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
            else:
                self.hlp_env.klist[self.cursor].trd = self.trdnow  # 上升趋势未跌破前低，趋势延续
                self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[self.cursor - 1].pre_trd
        elif self.trdnow == 'down' and self.hlp_env.klist[self.cursor].hl == 'l':
            if self.hlp_env.klist[self.cursor].high > self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-1]].high:
                # 同其镜像问题，该种情况也不会出现，一点突破前高点，当即确认低点，hl转为h
                self.trdchg.append(self.cursor)  # 下跌趋势，待判定低点，若期间突破前高，转为盘整
                self.hlp_env.klist[self.cursor].trd = 'consd'
                self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                self.trdnow = 'consd'
                printt(f"##`step_trd`，确认盘整趋势开启，位置{self.cursor}；"
                       f"之前趋势：down；确认条件：下跌趋势中低点未确认，突破前高点",
                       msg_mode=constant.RunMode.REPORT,
                       global_mode=constant.RUN_MODE)
            else:
                self.hlp_env.klist[self.cursor].trd = self.trdnow  # 下跌趋势，待判定低点，不突破前高，趋势保持
                self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[self.cursor - 1].pre_trd
        elif self.trdnow == 'down' and self.hlp_env.klist[self.cursor].hl == 'h':
            if self.hlp_env.klist[self.cursor].high > self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-1]].high:
                if self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-1]].low > \
                    self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-2]].low:
                    self.hlp_env.klist[self.cursor].trd = 'up'  # 下跌趋势，待判定高点，若期间突破前高，且低点已连续抬高，满足上涨条件
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'up'
                    self.trdchg.append(self.cursor)
                    printt(f"##`step_trd`，确认上涨趋势开启，位置{self.cursor}；"
                           f"之前趋势：down；确认条件：下跌趋势中待判定高点，期间突破前高点，低点已确认并连续抬高，追认上涨趋势",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                else:
                    self.hlp_env.klist[self.cursor].trd = 'consd'  #
                    # 下跌趋势，待判定高点，期间高点突破，下跌结束，但低点未连续抬高，不满足上涨条件，转为盘整
                    self.trdchg.append(self.cursor)
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'consd'
                    printt(f"##`step_trd`，确认盘整趋势开启，位置{self.cursor}；"
                           f"之前趋势：down；确认条件：下跌趋势中待判定高点，期间突破前高点，低点已确认但未连续抬高",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
            else:
                self.hlp_env.klist[self.cursor].trd = self.trdnow
                self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[self.cursor - 1].pre_trd
        elif self.trdnow == 'consd' and self.hlp_env.klist[self.cursor].hl == 'h':
            # 当前盘整趋势+待判定高点
            # NOTE：规范的盘整一般最终都可划归为两种形态
            # / 1.h1-l1-h2(h2>h1)-cursor(c<l1，跌破前低)，由上涨趋势转化而来
            # / 2.l1-h1-l2(l2<l1)-cursor(c>h1，突破前高)，由下跌趋势转化而来
            # 脱离盘整的判定有2种：
            # / 1.在step_trdmax中通过时间超跌超涨非结构性的转入上涨或下跌
            # / 2.在高点或低点被确认的K线上可能出现新状态

            if self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-1]].hl_confirmed == self.cursor:
                # 在当前K上确认最近低点，转入h
                if ((self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-1]].high >
                    self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-2]].high) and
                    (self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-1]].low >
                    self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-2]].low)):
                    # 通过确认低点后可以马上结束盘整的模式-->h1-l1-h2(h2>h1)-l2(确认，l2>l1)
                    self.hlp_env.klist[self.cursor].trd = 'up'
                    self.trdchg.append(self.cursor)
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'up'
                    printt(f"##`step_trd`，确认上涨趋势开启，位置{self.cursor}；"
                           f"之前趋势：{self.hlp_env.klist[self.cursor].pre_trd}；"
                           f"确认条件：低点被确认后立即符合上涨形态",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                else:
                    self.hlp_env.klist[self.cursor].trd = self.trdnow
                    self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[
                        self.cursor - 1].pre_trd
            else:
                self.hlp_env.klist[self.cursor].trd = self.trdnow
                self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[
                    self.cursor - 1].pre_trd

        elif self.trdnow == 'consd' and self.hlp_env.klist[self.cursor].hl == 'l':
            if self.hlp_env.klist[
                self.hlp_env.klist[self.cursor].hpi[-1]].hl_confirmed == self.cursor:
                # 当前K上确认最近高点
                if ((self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-1]].high <
                    self.hlp_env.klist[self.hlp_env.klist[self.cursor].hpi[-2]].high) and
                    (self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-1]].low <
                    self.hlp_env.klist[self.hlp_env.klist[self.cursor].lpi[-2]].low)):
                    # 通过确认低点后可以马上结束盘整的模式-->l1-h1-l2(l2<l1)-h2(确认，h2<h1)
                    self.hlp_env.klist[self.cursor].trd = 'down'
                    self.trdchg.append(self.cursor)
                    self.hlp_env.klist[self.cursor].pre_trd = self.trdnow
                    self.trdnow = 'down'
                    printt(f"##`step_trd`，确认下跌趋势开启，位置{self.cursor}；"
                           f"之前趋势：{self.hlp_env.klist[self.cursor].pre_trd}；"
                           f"确认条件：高点被确认后立即符合下跌形态",
                           msg_mode=constant.RunMode.REPORT,
                           global_mode=constant.RUN_MODE)
                else:
                    self.hlp_env.klist[self.cursor].trd = self.trdnow
                    self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[
                        self.cursor - 1].pre_trd
            else:
                self.hlp_env.klist[self.cursor].trd = self.trdnow
                self.hlp_env.klist[self.cursor].pre_trd = self.hlp_env.klist[
                    self.cursor - 1].pre_trd
        self.cursor += 1

    def get_trend(self):
        self.init_trd()
        while self.cursor < len(self.hlp_env.klist):
            res = self.step_trdmax()
            if not res:
                self.step_trd()


