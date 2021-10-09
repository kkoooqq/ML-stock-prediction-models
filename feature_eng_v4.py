import bottleneck as bn
import datetime
import time
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif  # 特征选择方法库
import math
from sklearn.pipeline import Pipeline  # 导入Pipeline库
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
from feature_selector import *
import os
import matplotlib
from pylab import mpl

try:
    import talib
except:
    pass

import joblib
from scipy.ndimage.interpolation import shift
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

try:
    from jqdatasdk import *
except:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'SimHei'

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

fe_version = 4

def jq_auth():
    auth('', '')

def get_factor_pre_core(security_list, date_list, pbar, df_factor_pre, date):
    try:
        df_old = pd.DataFrame(index=security_list)

        # 所有股票某个时间区间行情
        # 带上预测的天数，用来计算pct_change
        trade_all = None
        while True:
            try:
                jq_auth()
                # [ look_back_period个回看区间(上一个周期) | look_back_period 个回看区间(当前周期) | look_forward_period个预测区间数据 ]
                trade_all = get_price(security=security_list, count=look_back_period * 2 + look_forward_period, end_date=date_list[date_list.index(date) + 1], fields=['open', 'close', 'low', 'high', 'volume', 'money', 'factor', 'high_limit', 'low_limit', 'avg', 'pre_close', 'paused'], frequency='daily', fq='pre', panel=False)
                logout()
                break
            except Exception as e:
                if e.args[0] == '您的账号最多只能开启 3 个链接':
                    time.sleep(1)
                else:
                    print(e)

        trade_all_values = trade_all.values

        # 市值表
        q = query(valuation.capitalization,
                  valuation.circulating_cap,
                  valuation.market_cap,
                  valuation.circulating_market_cap,
                  valuation.turnover_ratio,
                  valuation.pe_ratio,
                  valuation.pe_ratio_lyr,
                  valuation.pb_ratio,
                  valuation.ps_ratio,
                  valuation.pcf_ratio
                  ).filter(valuation.code.in_(security_list))

        valuation_all = None
        while True:
            try:
                jq_auth()
                # FIXME: 此API有1w条最大限制
                valuation_all = get_fundamentals_continuously(q, count=look_back_period, end_date=date, panel=False)
                logout()
                break
            except Exception as e:
                if e.args[0] == '您的账号最多只能开启 3 个链接':
                    time.sleep(1)
                else:
                    print(e)

        valuation_all_values = valuation_all.values

        # 资金流信息
        money_flow_all = None
        while True:
            try:
                jq_auth()
                money_flow_all = get_money_flow(security_list=security_list, count=look_back_period, end_date=date, fields=['sec_code', 'date', 'change_pct', 'net_amount_main', 'net_pct_main', 'net_amount_xl', 'net_pct_xl', 'net_amount_l', 'net_pct_l', 'net_amount_m', 'net_pct_m', 'net_amount_s', 'net_pct_s'])
                logout()
                break
            except Exception as e:
                if e.args[0] == '您的账号最多只能开启 3 个链接':
                    time.sleep(1)
                else:
                    print(e)

        money_flow_all_values = money_flow_all.values

        for security in tqdm(security_list):
            # trade_values构成：
            # [ look_back_period个回看区间(上一个周期) | look_back_period 个回看区间(当前周期) | look_forward_period个预测区间数据 ]
            trade_values = trade_all_values[trade_all_values[:, 1] == security]
            trade_values_forward = trade_values[-look_forward_period:]
            trade_values = trade_values[: -look_forward_period]  # 剩下： [ look_back_period个回看区间(上一个周期) | look_back_period 个回看区间(当前周期) ]
            trade_values_1 = trade_values[: look_back_period]
            trade_values = trade_values[look_back_period:]

            if trade_values.shape[0] != look_back_period:
                continue

            valuation_values = valuation_all_values[valuation_all_values[:, 1] == security]
            if valuation_values.shape[0] != look_back_period:
                continue

            money_flow_values = money_flow_all_values[money_flow_all_values[:, 0] == security]
            if money_flow_values.shape[0] != look_back_period:
                continue

            # 预测区间
            close_forward_values = trade_values_forward[:, 3].astype(np.float64)

            # 当前区间行情
            open_values = trade_values[:, 2].astype(np.float64)
            high_values = trade_values[:, 5].astype(np.float64)
            low_values = trade_values[:, 4].astype(np.float64)
            close_values = trade_values[:, 3].astype(np.float64)
            vol_values = trade_values[:, 6].astype(np.float64)
            vol_values[vol_values == 0] = 1
            amount_values = trade_values[:, 7].astype(np.float64)
            amount_values[amount_values == 0] = 1
            high_limit_values = trade_values[:, 9].astype(np.float64)
            low_limit_values = trade_values[:, 10].astype(np.float64)

            # 上一个区间行情
            open_values_1 = trade_values_1[:, 2].astype(np.float64)
            high_values_1 = trade_values_1[:, 5].astype(np.float64)
            low_values_1 = trade_values_1[:, 4].astype(np.float64)
            close_values_1 = trade_values_1[:, 3].astype(np.float64)
            vol_values_1 = trade_values_1[:, 6].astype(np.float64)
            vol_values_1[vol_values_1 == 0] = 1
            amount_values_1 = trade_values_1[:, 7].astype(np.float64)
            amount_values_1[amount_values_1 == 0] = 1
            high_limit_values_1 = trade_values_1[:, 9].astype(np.float64)
            low_limit_values_1 = trade_values_1[:, 10].astype(np.float64)

            # 市值表
            market_cap_values = valuation_values[:, 4].astype(np.float64)
            circulating_market_cap_values = valuation_values[:, 5].astype(np.float64)
            turnover_ratio_values = valuation_values[:, 6].astype(np.float64)
            turnover_ratio_values = np.nan_to_num(turnover_ratio_values, copy=True, nan=0.0, posinf=None, neginf=None)
            turnover_ratio_values[turnover_ratio_values == 0] = 1

            # 资金流信息
            net_amount_main_values = money_flow_values[:, 3].astype(np.float64)
            net_amount_m_values = money_flow_values[:, 9].astype(np.float64)
            net_amount_s_values = money_flow_values[:, 11].astype(np.float64)

            net_pct_main_values = money_flow_values[:, 4].astype(np.float64)
            net_pct_main_values = np.nan_to_num(net_pct_main_values, copy=True, nan=0.0, posinf=None, neginf=None)
            net_pct_main_values[net_pct_main_values == 0] = 1

            net_pct_m_values = money_flow_values[:, 10].astype(np.float64)
            net_pct_m_values = np.nan_to_num(net_pct_m_values, copy=True, nan=0.0, posinf=None, neginf=None)
            net_pct_m_values[net_pct_m_values == 0] = 1

            net_pct_s_values = money_flow_values[:, 12].astype(np.float64)
            net_pct_s_values = np.nan_to_num(net_pct_s_values, copy=True, nan=0.0, posinf=None, neginf=None)
            net_pct_s_values[net_pct_s_values == 0] = 1

            # pchg，用预测区间的最后一天，除以回看区间最后一天
            df_old.loc[security, 'pchg'] = close_forward_values[-1] / close_values[-1]

            df_old.loc[security, '收盘上涨天数/收盘下跌天数'] = bn.nansum(close_values > shift(close_values, 1, cval=np.NaN)) / bn.nansum(close_values < shift(close_values, 1, cval=np.NaN))
            df_old.loc[security, '股价高开天数/股价低开天数'] = bn.nansum(open_values > shift(close_values, 1, cval=np.NaN)) / bn.nansum(open_values < shift(close_values, 1, cval=np.NaN))

            # 区间开盘价振幅 = 最高开盘价与最低开盘价之比
            df_old.loc[security, '开盘价振幅'] = bn.nanmax(open_values) / bn.nanmin(open_values)

            # 区间收盘价振幅 = 最高收盘价与最低收盘价之比
            df_old.loc[security, '收盘价振幅'] = bn.nanmax(close_values) / bn.nanmin(close_values)

            df_old.loc[security, '当日收盘价与最高收盘价振幅'] = close_values[-1] / bn.nanmax(close_values)
            df_old.loc[security, '当日收盘价与最高收盘价天数'] = look_back_period - np.where(close_values == bn.nanmax(close_values))[0][0]
            df_old.loc[security, '当日收盘价与最低收盘价振幅'] = close_values[-1] / bn.nanmin(close_values)
            df_old.loc[security, '当日收盘价与最低收盘价天数'] = look_back_period - np.where(close_values == bn.nanmin(close_values))[0][0]

            # 最高价振幅 = 最大最高价与最小最高价之比
            df_old.loc[security, '最高价振幅'] = bn.nanmax(high_values) / bn.nanmin(high_values)

            # 最低价振幅 = 最大最低价与最小最低价之比
            df_old.loc[security, '最低价振幅'] = bn.nanmax(low_values) / bn.nanmin(low_values)

            # 最高价与最低价振幅 = 区间最高价与区间最低价之比
            df_old.loc[security, '最高价与最低价振幅'] = bn.nanmax(high_values) / bn.nanmin(low_values)
            df_old.loc[security, '每日最高价与最低价最大振幅'] = bn.nanmax(high_values / low_values)

            df_old.loc[security, '成交量上涨天数/成交量下跌天数'] = bn.nansum(vol_values > shift(vol_values, 1, cval=np.NaN)) / bn.nansum(vol_values < shift(vol_values, 1, cval=np.NaN))

            # 区间成交量振幅 = 最大成交量与最小成交量之比
            df_old.loc[security, '成交量振幅'] = bn.nanmax(vol_values) / bn.nanmin(vol_values[vol_values > 0])

            df_old.loc[security, '成交金额上涨天数/成交金额下跌天数'] = bn.nansum(amount_values > shift(amount_values, 1, cval=np.NaN)) / bn.nansum(amount_values < shift(amount_values, 1, cval=np.NaN))

            # 区间成交金额振幅 = 最大成交金额与最小成交金额之比
            df_old.loc[security, '成交金额振幅'] = bn.nanmax(amount_values) / bn.nanmin(amount_values[amount_values > 0])

            # 区间倒数3日成交量之和 / 前面7日成交量之和
            df_old.loc[security, '倒数3日成交量之和/前面成交量之和'] = bn.nansum(vol_values[-3:]) / bn.nansum(vol_values[: -3])

            # 区间倒数3日平均成交量 / 前面7日平均成交量
            df_old.loc[security, '倒数3日平均成交量/前面平均成交量'] = bn.nanmean(vol_values[-3:]) / bn.nanmean(vol_values[: -3])

            df_old.loc[security, '换手率上涨天数/换手率下跌天数'] = bn.nansum(turnover_ratio_values > shift(turnover_ratio_values, 1, cval=np.NaN)) / bn.nansum(turnover_ratio_values < shift(turnover_ratio_values, 1, cval=np.NaN))

            #因子20 = 区间换手率振幅 = 最高换手率/最低换手率
            df_old.loc[security, '换手率振幅'] = bn.nanmax(turnover_ratio_values) / bn.nanmin(turnover_ratio_values[turnover_ratio_values > 0])

            #因子21 = 区间倒数3日换手率之和 / 前面7日换手率之和
            df_old.loc[security, '倒数3日换手率之和/前面换手率之和'] = bn.nansum(turnover_ratio_values[-3:]) / bn.nansum(turnover_ratio_values[: -3])

            #因子22 = 区间倒数3日平均换手率 / 前面7日平均换手率
            df_old.loc[security, '倒数3日平均换手率/前面平均换手率'] = bn.nanmean(turnover_ratio_values[-3:]) / bn.nanmean(turnover_ratio_values[: -3])

            #因子23 = 大阳线天数 = 统计区间单日上涨>=5%的天数
            df_old.loc[security, '大阳线天数'] = bn.nansum(close_values / open_values > 1.05) 

            #因子24 = 大阴线天数 = 统计区间单日下跌>=5%的天数
            df_old.loc[security, '大阴线天数'] = bn.nansum(close_values / open_values < 0.95) 

            #因子25 = 涨停板 = 的天数
            df_old.loc[security, '涨停天数'] = bn.nansum(close_values == high_limit_values) 

            #因子26 = 跌停板 = 的天数
            df_old.loc[security, '跌停天数'] = bn.nansum(close_values == low_limit_values) 

            #因子27 = 涨跌停板振幅 = 最大单日涨幅 / 最大单日跌幅
            df_old.loc[security, '涨跌停板振幅'] = bn.nanmax(close_values / shift(close_values, 1, cval=np.NaN)) / bn.nanmin(close_values / shift(close_values, 1, cval=np.NaN))

            #因子28 = 股价连续上涨天数
            max_increase_count = 0
            increase_count = 0

            for n in range(1, len(close_values)):
                if close_values[n] > close_values[n - 1]:
                    increase_count += 1
                    max_increase_count = max(increase_count, max_increase_count)
                else:
                    increase_count = 0

            df_old.loc[security, '股价连续上涨最大天数'] = max_increase_count 

            #因子29 = 股价连续下跌天数
            max_decrease_count = 0
            decrease_count = 0

            for n in range(1, len(close_values)):
                if close_values[n] < close_values[n - 1]:
                    decrease_count += 1
                    max_decrease_count = max(decrease_count, max_decrease_count)
                else:
                    decrease_count = 0

            df_old.loc[security, '股价连续下跌最大天数'] = max_decrease_count 

            #因子30 = 区间倒数3日收盘均价 / 前面7日收盘均价
            df_old.loc[security, '倒数3日收盘均价/前面收盘均价'] = bn.nanmean(close_values[-3:]) / bn.nanmean(close_values[: -3])
            df_old.loc[security, '倒数3日收盘均价/收盘均价'] = bn.nanmean(close_values[-3:]) / bn.nanmean(close_values)

            # 因子31 市值增长率 倒数最后一个交易日市值/第一个交易日市值
            df_old.loc[security, '倒数最后一个交易日市值/第一个交易日市值'] = market_cap_values[-1] / market_cap_values[0]

            df_old.loc[security, '市值上涨天数/市值下跌天数'] = bn.nansum(market_cap_values > shift(market_cap_values, 1, cval=np.NaN)) / bn.nansum(market_cap_values < shift(market_cap_values, 1, cval=np.NaN))

            # 因子34 倒数3日平均市值/前面7日平均市值
            df_old.loc[security, '倒数3日平均市值/前面平均市值'] = bn.nanmean(market_cap_values[-3:]) / bn.nanmean(market_cap_values[: -3])
            df_old.loc[security, '倒数3日平均市值/平均市值'] = bn.nanmean(market_cap_values[-3:]) / bn.nanmean(market_cap_values)

            df_old.loc[security, '大单资金净流入天数/大单资金净流出天数'] = bn.nansum(net_pct_main_values > 0) / bn.nansum(net_pct_main_values < 0)

            # 因子37 统计区间资金连续流入天数
            max_increase_count = 0
            increase_count = 0

            for net_pct_main_value in net_pct_main_values:
                if net_pct_main_value > 0:
                    increase_count += 1
                    max_increase_count = max(increase_count, max_increase_count)
                else:
                    increase_count = 0

            df_old.loc[security, '资金连续流入天数'] = max_increase_count 

            # 因子38 统计区间资金连续流出天数
            max_decrease_count = 0
            decrease_count = 0

            for net_pct_main_value in net_pct_main_values:
                if net_pct_main_value < 0:
                    decrease_count += 1
                    max_decrease_count = max(decrease_count, max_decrease_count)
                else:
                    decrease_count = 0

            df_old.loc[security, '资金连续流出天数'] = max_decrease_count 

            # 因子39 统计区间资金净流入大单占比
            df_old.loc[security, '主力净流入占比之和'] = bn.nansum(net_pct_main_values[net_pct_main_values > 0])

            # 因子40 统计区间资金净流出大单占比
            df_old.loc[security, '主力净流出占比之和'] = bn.nansum(net_pct_main_values[net_pct_main_values < 0])

            df_old.loc[security, '主力占比平均值'] = bn.nanmean(net_pct_main_values)
            df_old.loc[security, '中单占比平均值'] = bn.nanmean(net_pct_m_values)
            df_old.loc[security, '小单占比平均值'] = bn.nanmean(net_pct_s_values)
            df_old.loc[security, '主力占流通市值'] = bn.nansum(net_pct_main_values / circulating_market_cap_values)
            df_old.loc[security, '中单占流通市值'] = bn.nansum(net_pct_m_values / circulating_market_cap_values)
            df_old.loc[security, '小单占流通市值'] = bn.nansum(net_pct_s_values / circulating_market_cap_values)

            # df_old.loc[security, '回看区间主力占比'] = bn.nansum(net_amount_main_values) / (bn.nansum(net_amount_main_values) + bn.nansum(net_amount_m_values) + bn.nansum(net_amount_s_values))
            # df_old.loc[security, '回看区间中单占比'] = bn.nansum(net_amount_m_values) / (bn.nansum(net_amount_main_values) + bn.nansum(net_amount_m_values) + bn.nansum(net_amount_s_values))
            # df_old.loc[security, '回看区间小单占比'] = bn.nansum(net_amount_s_values) / (bn.nansum(net_amount_main_values) + bn.nansum(net_amount_m_values) + bn.nansum(net_amount_s_values))

            # df_old.loc[security, '倒数3日主力占比'] = bn.nansum(net_amount_main_values[-3:]) / (bn.nansum(net_amount_main_values) + bn.nansum(net_amount_m_values) + bn.nansum(net_amount_s_values))
            # df_old.loc[security, '倒数3日中单占比'] = bn.nansum(net_amount_m_values[-3:]) / (bn.nansum(net_amount_main_values) + bn.nansum(net_amount_m_values) + bn.nansum(net_amount_s_values))
            # df_old.loc[security, '倒数3日小单占比'] = bn.nansum(net_amount_s_values[-3:]) / (bn.nansum(net_amount_main_values) + bn.nansum(net_amount_m_values) + bn.nansum(net_amount_s_values))

            df_old.loc[security, '倒数3日主力平均值'] = bn.nanmean(net_amount_main_values[-3:])
            df_old.loc[security, '倒数3日中单平均值'] = bn.nanmean(net_amount_m_values[-3:])
            df_old.loc[security, '倒数3日小单平均值'] = bn.nanmean(net_amount_s_values[-3:])
            df_old.loc[security, '倒数3日主力占流通市值'] = bn.nansum(net_amount_main_values[-3:] / circulating_market_cap_values[-3:])
            df_old.loc[security, '倒数3日中单占流通市值'] = bn.nansum(net_amount_m_values[-3:] / circulating_market_cap_values[-3:])
            df_old.loc[security, '倒数3日小单占流通市值'] = bn.nansum(net_amount_s_values[-3:] / circulating_market_cap_values[-3:])

            # 因子41 判断区间首个交易日开盘价是否高于区间前面10个交易日收盘价均价，是为1否则-1
            df_old.loc[security, '首个交易日开盘价牛逼10'] = 1 if open_values[0] > bn.nanmean(close_values_1[-10:]) else -1

            # 因子42 判断区间首个交易日开盘价是否高于区间前面5个交易日收盘价均价，是为1否则-1
            df_old.loc[security, '首个交易日开盘价牛逼5'] = 1 if open_values[0] > bn.nanmean(close_values_1[-5:]) else -1

            # 因子43 判断区间最后一个交易日收盘价是否高于区间内10个交易日收盘价均价，是为1否则-1
            df_old.loc[security, '最后交易日收盘价牛逼10'] = 1 if close_values[-1] > bn.nanmean(close_values[-10:]) else -1

            # 特征44：个股净流入资金/板块净流入资金*100%

            # 特征45：区间内北上资金累计净流入 / 区间内北上资金累计净流出*100%

            # 因子：talib形态
            # https://github.com/HuaRongSAO/talib-document/blob/master/func_groups/pattern_recognition.md
            df_old.loc[security, '形态:两只乌鸦'] = talib.CDL2CROWS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三只乌鸦'] = talib.CDL3BLACKCROWS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三内部上涨和下跌'] = talib.CDL3INSIDE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三线打击'] = talib.CDL3LINESTRIKE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三外部上涨和下跌'] = talib.CDL3OUTSIDE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:南方三星'] = talib.CDL3STARSINSOUTH(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三个白兵'] = talib.CDL3WHITESOLDIERS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:弃婴'] = talib.CDLABANDONEDBABY(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:大敌当前'] = talib.CDLADVANCEBLOCK(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:捉腰带线'] = talib.CDLBELTHOLD(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:脱离'] = talib.CDLBREAKAWAY(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:收盘缺影线'] = talib.CDLCLOSINGMARUBOZU(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:藏婴吞没'] = talib.CDLCONCEALBABYSWALL(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:反击线'] = talib.CDLCOUNTERATTACK(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:乌云压顶'] = talib.CDLDARKCLOUDCOVER(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:十字'] = talib.CDLDOJI(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:十字星'] = talib.CDLDOJISTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:蜻蜓十字/T形十字'] = talib.CDLDRAGONFLYDOJI(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:吞噬模式'] = talib.CDLENGULFING(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:十字暮星'] = talib.CDLEVENINGDOJISTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:暮星'] = talib.CDLEVENINGSTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:向上/下跳空并列阳线'] = talib.CDLGAPSIDESIDEWHITE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:墓碑十字/倒T十字'] = talib.CDLGRAVESTONEDOJI(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:锤头'] = talib.CDLHAMMER(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:上吊线'] = talib.CDLHANGINGMAN(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:母子线'] = talib.CDLHARAMI(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:十字孕线'] = talib.CDLHARAMICROSS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:风高浪大线'] = talib.CDLHIGHWAVE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:陷阱'] = talib.CDLHIKKAKE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:修正陷阱'] = talib.CDLHIKKAKEMOD(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:家鸽'] = talib.CDLHOMINGPIGEON(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三胞胎乌鸦'] = talib.CDLIDENTICAL3CROWS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:颈内线'] = talib.CDLINNECK(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:倒锤头'] = talib.CDLINVERTEDHAMMER(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:反冲形态'] = talib.CDLKICKING(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:由较长缺影线决定的反冲形态'] = talib.CDLKICKINGBYLENGTH(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:梯底'] = talib.CDLLADDERBOTTOM(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:长脚十字'] = talib.CDLLONGLEGGEDDOJI(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:长蜡烛'] = talib.CDLLONGLINE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:光头光脚/缺影线'] = talib.CDLMARUBOZU(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:相同低价'] = talib.CDLMATCHINGLOW(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:铺垫'] = talib.CDLMATHOLD(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:十字晨星'] = talib.CDLMORNINGDOJISTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:晨星'] = talib.CDLMORNINGSTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:颈上线'] = talib.CDLONNECK(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:刺透形态'] = talib.CDLPIERCING(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:黄包车夫'] = talib.CDLRICKSHAWMAN(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:上升/下降三法'] = talib.CDLRISEFALL3METHODS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:分离线'] = talib.CDLSEPARATINGLINES(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:射击之星'] = talib.CDLSHOOTINGSTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:短蜡烛'] = talib.CDLSHORTLINE(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:纺锤'] = talib.CDLSPINNINGTOP(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:停顿形态'] = talib.CDLSTALLEDPATTERN(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:条形三明治'] = talib.CDLSTICKSANDWICH(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:探水竿'] = talib.CDLTAKURI(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:跳空并列阴阳线'] = talib.CDLTASUKIGAP(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:插入'] = talib.CDLTHRUSTING(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:三星'] = talib.CDLTRISTAR(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:奇特三河床'] = talib.CDLUNIQUE3RIVER(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:向上跳空的两只乌鸦'] = talib.CDLUPSIDEGAP2CROWS(open_values, high_values, low_values, close_values)[-1]
            df_old.loc[security, '形态:上升/下降跳空三法'] = talib.CDLXSIDEGAP3METHODS(open_values, high_values, low_values, close_values)[-1]

        df_factor_pre[date] = df_old
    # pbar.update(len(df_factor_pre.keys()))
    except Exception as e:
        print('!!!!!!!!!!!!! error:' + e)

    print(f'{date} done')


def get_factor_pre(security_list, date_list):
    if os.path.exists(f'feature_eng_v{fe_version}.joblib'):
        df_factor_pre = joblib.load(f'feature_eng_v{fe_version}.joblib')
    else:
        all_task = []
        executor = ThreadPoolExecutor(max_workers=32)
        pbar = None  # tqdm(total=len(date_list[: -1]))

        df_factor_pre = {}

        for date in date_list[: -1]:
            all_task.append(executor.submit(get_factor_pre_core, security_list, date_list, pbar, df_factor_pre, date))
            # get_factor_pre_core(pbar, df_factor_pre, date)

        wait(all_task, return_when=ALL_COMPLETED)
        joblib.dump(df_factor_pre, f'feature_eng_v{fe_version}.joblib', protocol=4)

    # print(df_factor_pre[list(df_factor_pre.keys())[0]].head(20))

    # write_file('初始因子值数据表.csv', factor_data.to_csv(), append=False)

    return df_factor_pre


if __name__ == "__main__":
    from tqdm import tqdm

    # 回看区间，预测期间
    look_back_period = 15
    look_forward_period = 3

    try:
        os.chdir('bin/2.模型')
    except Exception as e:
        pass

    date_list = ['2019-01-02', '2019-01-07', '2019-01-10', '2019-01-15', '2019-01-18', '2019-01-23', '2019-01-28', '2019-01-31', '2019-02-12', '2019-02-15', '2019-02-20', '2019-02-25', '2019-02-28', '2019-03-05', '2019-03-08', '2019-03-13', '2019-03-18', '2019-03-21', '2019-03-26', '2019-03-29', '2019-04-03', '2019-04-09', '2019-04-12', '2019-04-17', '2019-04-22', '2019-04-25', '2019-04-30', '2019-05-08', '2019-05-13', '2019-05-16', '2019-05-21', '2019-05-24', '2019-05-29', '2019-06-03', '2019-06-06', '2019-06-12', '2019-06-17', '2019-06-20', '2019-06-25', '2019-06-28', '2019-07-03', '2019-07-08', '2019-07-11', '2019-07-16', '2019-07-19', '2019-07-24', '2019-07-29', '2019-08-01', '2019-08-06', '2019-08-09', '2019-08-14', '2019-08-19', '2019-08-22', '2019-08-27', '2019-08-30', '2019-09-04', '2019-09-09', '2019-09-12', '2019-09-18', '2019-09-23', '2019-09-26', '2019-10-08', '2019-10-11', '2019-10-16', '2019-10-21', '2019-10-24', '2019-10-29', '2019-11-01', '2019-11-06', '2019-11-11', '2019-11-14', '2019-11-19', '2019-11-22', '2019-11-27', '2019-12-02', '2019-12-05', '2019-12-10', '2019-12-13', '2019-12-18', '2019-12-23', '2019-12-26', '2019-12-31', '2020-01-06',
                 '2020-01-09', '2020-01-14', '2020-01-17', '2020-01-22', '2020-02-04', '2020-02-07', '2020-02-12', '2020-02-17', '2020-02-20', '2020-02-25', '2020-02-28', '2020-03-04', '2020-03-09', '2020-03-12', '2020-03-17', '2020-03-20', '2020-03-25', '2020-03-30', '2020-04-02', '2020-04-08', '2020-04-13', '2020-04-16', '2020-04-21', '2020-04-24', '2020-04-29', '2020-05-07', '2020-05-12', '2020-05-15', '2020-05-20', '2020-05-25', '2020-05-28', '2020-06-02', '2020-06-05', '2020-06-10', '2020-06-15', '2020-06-18', '2020-06-23', '2020-06-30', '2020-07-03', '2020-07-08', '2020-07-13', '2020-07-16', '2020-07-21', '2020-07-24', '2020-07-29', '2020-08-03', '2020-08-06', '2020-08-11', '2020-08-14', '2020-08-19', '2020-08-24', '2020-08-27', '2020-09-01', '2020-09-04', '2020-09-09', '2020-09-14', '2020-09-17', '2020-09-22', '2020-09-25', '2020-09-30', '2020-10-13', '2020-10-16', '2020-10-21', '2020-10-26', '2020-10-29', '2020-11-03', '2020-11-06', '2020-11-11', '2020-11-16', '2020-11-19', '2020-11-24', '2020-11-27', '2020-12-02', '2020-12-07', '2020-12-10', '2020-12-15', '2020-12-18', '2020-12-23', '2020-12-28', '2020-12-31', '2021-01-06', '2021-01-11', '2021-01-14']
    security_list = ['000001.XSHE', '000002.XSHE', '000063.XSHE', '000066.XSHE', '000069.XSHE', '000100.XSHE', '000157.XSHE', '000166.XSHE', '000333.XSHE', '000338.XSHE', '000425.XSHE', '000538.XSHE', '000568.XSHE', '000596.XSHE', '000625.XSHE', '000627.XSHE', '000651.XSHE', '000656.XSHE', '000661.XSHE', '000671.XSHE', '000703.XSHE', '000708.XSHE', '000723.XSHE', '000725.XSHE', '000728.XSHE', '000768.XSHE', '000776.XSHE', '000783.XSHE', '000786.XSHE', '000858.XSHE', '000860.XSHE', '000876.XSHE', '000895.XSHE', '000938.XSHE', '000961.XSHE', '000963.XSHE', '000977.XSHE', '001979.XSHE', '002001.XSHE', '002007.XSHE', '002008.XSHE', '002024.XSHE', '002027.XSHE', '002032.XSHE', '002044.XSHE', '002049.XSHE', '002050.XSHE', '002120.XSHE', '002129.XSHE', '002142.XSHE', '002146.XSHE', '002153.XSHE', '002157.XSHE', '002179.XSHE', '002202.XSHE', '002230.XSHE', '002236.XSHE', '002241.XSHE', '002252.XSHE', '002271.XSHE', '002304.XSHE', '002311.XSHE', '002352.XSHE', '002371.XSHE', '002384.XSHE', '002410.XSHE', '002414.XSHE', '002415.XSHE', '002422.XSHE', '002456.XSHE', '002460.XSHE', '002463.XSHE', '002475.XSHE', '002493.XSHE', '002508.XSHE', '002555.XSHE', '002558.XSHE', '002594.XSHE', '002600.XSHE', '002601.XSHE', '002602.XSHE', '002607.XSHE', '002624.XSHE', '002673.XSHE', '002714.XSHE', '002736.XSHE', '002739.XSHE', '002773.XSHE', '002812.XSHE', '002821.XSHE', '002841.XSHE', '002916.XSHE', '002938.XSHE', '300003.XSHE', '300014.XSHE', '300015.XSHE', '300033.XSHE', '300059.XSHE', '300122.XSHE', '300124.XSHE', '300136.XSHE', '300142.XSHE', '300144.XSHE', '300347.XSHE', '300408.XSHE', '300413.XSHE', '300433.XSHE', '300498.XSHE', '300529.XSHE', '300601.XSHE', '300628.XSHE', '300676.XSHE', '600000.XSHG', '600004.XSHG', '600009.XSHG', '600010.XSHG', '600011.XSHG', '600015.XSHG', '600016.XSHG', '600018.XSHG', '600019.XSHG', '600025.XSHG', '600027.XSHG', '600028.XSHG', '600029.XSHG', '600030.XSHG', '600031.XSHG', '600036.XSHG', '600048.XSHG', '600050.XSHG', '600061.XSHG', '600066.XSHG', '600068.XSHG', '600085.XSHG', '600104.XSHG', '600109.XSHG', '600111.XSHG', '600115.XSHG',
                     '600118.XSHG', '600161.XSHG', '600176.XSHG', '600177.XSHG', '600183.XSHG', '600196.XSHG', '600208.XSHG', '600233.XSHG', '600271.XSHG', '600276.XSHG', '600297.XSHG', '600299.XSHG', '600309.XSHG', '600332.XSHG', '600340.XSHG', '600346.XSHG', '600352.XSHG', '600362.XSHG', '600369.XSHG', '600383.XSHG', '600390.XSHG', '600406.XSHG', '600436.XSHG', '600438.XSHG', '600482.XSHG', '600487.XSHG', '600489.XSHG', '600498.XSHG', '600519.XSHG', '600522.XSHG', '600547.XSHG', '600570.XSHG', '600584.XSHG', '600585.XSHG', '600588.XSHG', '600600.XSHG', '600606.XSHG', '600637.XSHG', '600655.XSHG', '600660.XSHG', '600690.XSHG', '600703.XSHG', '600705.XSHG', '600741.XSHG', '600745.XSHG', '600760.XSHG', '600763.XSHG', '600795.XSHG', '600809.XSHG', '600837.XSHG', '600845.XSHG', '600848.XSHG', '600872.XSHG', '600886.XSHG', '600887.XSHG', '600893.XSHG', '600900.XSHG', '600919.XSHG', '600926.XSHG', '600958.XSHG', '600998.XSHG', '600999.XSHG', '601006.XSHG', '601009.XSHG', '601012.XSHG', '601021.XSHG', '601066.XSHG', '601088.XSHG', '601100.XSHG', '601108.XSHG', '601111.XSHG', '601117.XSHG', '601138.XSHG', '601155.XSHG', '601166.XSHG', '601169.XSHG', '601186.XSHG', '601198.XSHG', '601211.XSHG', '601216.XSHG', '601225.XSHG', '601229.XSHG', '601231.XSHG', '601238.XSHG', '601288.XSHG', '601318.XSHG', '601328.XSHG', '601336.XSHG', '601360.XSHG', '601377.XSHG', '601390.XSHG', '601398.XSHG', '601555.XSHG', '601577.XSHG', '601600.XSHG', '601601.XSHG', '601607.XSHG', '601618.XSHG', '601628.XSHG', '601633.XSHG', '601668.XSHG', '601669.XSHG', '601688.XSHG', '601727.XSHG', '601766.XSHG', '601788.XSHG', '601800.XSHG', '601808.XSHG', '601818.XSHG', '601838.XSHG', '601857.XSHG', '601872.XSHG', '601877.XSHG', '601878.XSHG', '601881.XSHG', '601888.XSHG', '601899.XSHG', '601901.XSHG', '601919.XSHG', '601933.XSHG', '601939.XSHG', '601985.XSHG', '601988.XSHG', '601989.XSHG', '601990.XSHG', '601998.XSHG', '603019.XSHG', '603156.XSHG', '603160.XSHG', '603259.XSHG', '603288.XSHG', '603369.XSHG', '603501.XSHG', '603658.XSHG', '603799.XSHG', '603833.XSHG', '603899.XSHG', '603986.XSHG', '603993.XSHG']

    get_factor_pre(security_list, date_list)
