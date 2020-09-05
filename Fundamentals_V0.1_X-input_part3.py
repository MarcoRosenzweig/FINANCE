#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


OUTPUT_FOLDER = "Option_Weekly_csv"
import os
try:
    os.mkdir(OUTPUT_FOLDER)
except FileExistsError:
    pass

def fundamentals (Company_Names, output_folder=OUTPUT_FOLDER, export_table=True):

    C_N_MarketCap = []
    C_N_Beta = []

    for x in Company_Names:
        try:
            ticker = yf.Ticker(x)
            info_dic = ticker.info
            var1 = info_dic['marketCap']/1000000000
            var2 = info_dic['beta']
            print(x, " Downloaded")
            C_N_MarketCap.append(var1)
            C_N_Beta.append(var2)

        except IndexError:
            C_N_MarketCap.append("I E")
            C_N_Beta.append("I E")
            continue
        except TypeError:
            C_N_MarketCap.append("T E")
            C_N_Beta.append("T E")
            continue

    year_pct_change_list = []
    data_vol_list = []

    for x in Company_Names:
        todaydate = datetime.date.today()
        data = yf.download(x, start= todaydate - datetime.timedelta(days=30))
        data_c = data["Close"]
        data_beg = data_c[-7]
        data_end = data_c[-1]
        #Calculate Return
        year_pct_change = (data_end-data_beg)/data_beg*100
        year_pct_change_list.append(year_pct_change)
        todaydate = datetime.date.today()

        #Calculate Volatility
        data_change = np.diff(data_c)
        data_pctchange = data_change/data_c[:-1]*100
        data_vol = data_pctchange.std()
        data_vol_list.append(data_vol)

    Fundamentals_X_input = pd.DataFrame({"Tickers":Company_Names,"MarketCap":C_N_MarketCap,"Beta":C_N_Beta,                                          "1W %Change":year_pct_change_list, "30day Volatility":data_vol_list})

    data = yf.download("SPY", start= todaydate - datetime.timedelta(days=7))
    benchmark = data["Close"]
    benchmark_beg = benchmark[0]
    benchmark_end = benchmark[-1]
    #benchmark_pct_change as effect on option prediction & predicted price
    benchmark_1week_pct_change = (benchmark_end-benchmark_beg)/benchmark_beg*100
    benchmark_pct_change_list = [benchmark_pct_change]*np.shape(year_pct_change_list)[0]

    Fundamentals_X_input["1W Ticker vs Benchmark Change"] = Fundamentals_X_input["1W %Change"] -    benchmark_pct_change_list

    todaydate = datetime.date.today()
    today_str = str(todaydate).replace("-", "_")
    try:
        folder = os.path.join(output_folder, today_str)
        os.mkdir(folder)
    except FileExistsError:
        pass

    table_name = os.path.join(folder, "Company_Fundamentals_{}.csv".format(today_str))
    if export_table:
        Fundamentals_X_input.to_csv(table_name)
        print("Exported: {}".format(table_name))

    return Fundamentals_X_input
