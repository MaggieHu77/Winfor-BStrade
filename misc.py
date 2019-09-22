# -*- coding:utf-8 -*-
# ! python3
from os import path
from typing import List
from openpyxl import load_workbook
from xlrd import open_workbook


def printt(msg: str, msg_mode: int, global_mode: int):
    """
    控制是否打印内容，主要用于兼容开发和运行使用，只有信息本身级别不低于全局打印级别，内容才被打印
    级别越低，打印内容越详细
    :param msg: str，需要打印的内容
    :param msg_mode: 待打印内容自身mode级别
    :param global_mode: 全局打印级别
    """
    if msg_mode >= global_mode:
        if msg_mode == 1:
            prefix = "\033[0m"
        elif msg_mode == 2:
            prefix = "\033[34m"
        elif msg_mode == 3:
            prefix = "\033[35m"
        else:
            prefix = "\033[31m"
        msg = prefix + msg
        print(msg)


def read_file(f: str) -> List[str]:
    """
    从文件地址中读取需要回测的股票代码或指数。

    :param f: 文件的绝对路径
    :return: 需要回测的股票代码字符串list
    """
    b_name = path.basename(f)
    b_type = b_name.split(".")[-1]
    codes = []
    if b_type == "txt":
        for code in open(f):
            code = code.strip("\n|;|,|/|")
            codes.append(code)
    elif b_type == "xlsx":
        wb = load_workbook(f)
        sheet = wb.active
        for code in list(sheet.columns)[0]:
            if "." in code.value:
                codes.append(code.value)
    elif b_type == "xls":
        wb = open_workbook(f)
        sheet = wb.sheet_by_index(0)
        for code in sheet.col_values(0):
            if "." in code:
                codes.append(code)
    else:
        print("Error: 不支持的文件格式 %s， 请选择.txt, .xls .xlsx之一" % b_type)
    return codes