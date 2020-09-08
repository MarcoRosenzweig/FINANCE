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
            C_N_MarketCap.append(0)
            C_N_Beta.append(0)
            continue
        except TypeError:
            C_N_MarketCap.append(0)
            C_N_Beta.append(0)
            continue

    year_pct_change_list = []
    data_vol_list = []
    moving_averages_list = []
    price_vs_ma_list = []
    mean_return_list = []
    var_list = []
    sharpe_ratio_list = []
    todaydate = datetime.date.today()

    five_year_bond = yf.download("^FVX", start= todaydate - datetime.timedelta(days=3))
    thirteen_week_bond = yf.download("^IRX", start= todaydate - datetime.timedelta(days=3))
    five_year_bond = five_year_bond["Close"]
    five_year_bond_p = five_year_bond[-1]
    thirteen_week_bond = thirteen_week_bond["Close"]
    thirteen_week_bond_p = thirteen_week_bond[-1]

    for x in Company_Names:

        data = yf.download(x, start= todaydate - datetime.timedelta(days=300))
        data_c = data["Close"]
        data_beg = data_c[-7]
        data_end = data_c[-1]
        data_beg_sharpe = data_c[0]
        data_end_sharpe = data_c[-1]
        data_ma_prices = np.sum(data_c[-50:])
        data_ma_days = np.shape(data_c[-50:])
        #Calculate Return
        year_pct_change = (data_end-data_beg)/data_beg*100
        year_pct_change_list.append(year_pct_change)
        #Calculate Moving Average
        moving_average = float(data_ma_prices / data_ma_days)
        moving_averages_list.append(moving_average)
        price_vs_ma_pct = (moving_average - data_end)/data_end*100
        price_vs_ma_list.append(price_vs_ma_pct)
        #Calculate Volatility
        data_change = np.diff(data_c[-52:-1])
        data_pctchange = data_change/data_c[-51:-1]*100
        data_vol = data_pctchange.std()
        data_vol_list.append(data_vol)
        #Calculate mean daily Return
        data_change = np.diff(data_c)
        data_pctchange = data_change / data_c[:-1]*100
        data_mean = round(data_pctchange.mean(),4)
        mean_return_list.append(data_mean)
        #calculate VaR
        VaR = round(data_mean - data_pctchange.std()*2,2)
        var_list.append(VaR)
        #Calculate Sharpe Ratio
        year_pct_change_sharpe = (data_end_sharpe-data_beg_sharpe)/data_beg_sharpe*100

        data_change_sharpe = np.diff(data_c)
        data_pctreturn_sharpe = data_change_sharpe/data_c[:-1]*100
        data_sig_sharpe = data_pctreturn_sharpe.std()

        bond_return_avg = (five_year_bond_p + thirteen_week_bond_p)/2
        bond_return_avg

        sharpe_ratio = (year_pct_change_sharpe-bond_return_avg)/data_sig_sharpe
        sharpe_ratio_list.append(sharpe_ratio)

    Fundamentals_X_input = pd.DataFrame({"Tickers":Company_Names,"MarketCap":C_N_MarketCap,"Beta":C_N_Beta, "50 Day MA": moving_averages_list,\
    "MA to TP %":price_vs_ma_list, "1W %Change":year_pct_change_list, "30day Vol":data_vol_list,\
    "Avg Daily %Return":mean_return_list, "95% VaR": var_list, "Sharpe Ratio": sharpe_ratio_list})

    data = yf.download("SPY", start= todaydate - datetime.timedelta(days=7))
    benchmark = data["Close"]
    benchmark_beg = benchmark[0]
    benchmark_end = benchmark[-1]
    #benchmark_pct_change as effect on option prediction & predicted price
    benchmark_1week_pct_change = (benchmark_end-benchmark_beg)/benchmark_beg*100
    benchmark_pct_change_list = [benchmark_1week_pct_change]*np.shape(year_pct_change_list)[0]

    Fundamentals_X_input["1W Ticker vs Bm %Change"] = Fundamentals_X_input["1W %Change"] -    benchmark_pct_change_list

    plt.figure(figsize=(18,11))
    plt.grid()
    plt.title("Beta(X) vs Sharpe Ratio(Y) - Relationship between Systematic Risk & Risk Adjusted Return")
    for x in Fundamentals_X_input["Tickers"]:
        Fundamentals_X_input_graph = Fundamentals_X_input.loc[lambda Fundamentals_X_input: Fundamentals_X_input['Tickers']== x, :]
        Fundamentals_X_input_graph_reset = Fundamentals_X_input_graph.reset_index(drop=True)

        plt.scatter(Fundamentals_X_input_graph_reset["Beta"],\
                    Fundamentals_X_input_graph_reset["Sharpe Ratio"])
        plt.annotate(x,(Fundamentals_X_input_graph_reset["Beta"],Fundamentals_X_input_graph_reset["Sharpe Ratio"]),\
                     textcoords="offset points", xytext=(7,2), ha='center')
        plt.xlabel("Beta - higher beta = higher systematic risk, Market Beta is always 1")
        plt.ylabel("Sharpe Ratio - higher sharpe ratio = higher return per unit of risk, hence seek max Sharpe Ratio")

    plt.figure(figsize=(18,11))
    plt.grid()
    plt.title("Avg Daily %Return(X) vs 95% VaR(Y) - Relationship between Daily Return & Maximum Daily Drawdown")
    for x in Fundamentals_X_input["Tickers"]:
        Fundamentals_X_input_graph = Fundamentals_X_input.loc[lambda Fundamentals_X_input: Fundamentals_X_input['Tickers']== x, :]
        Fundamentals_X_input_graph_reset = Fundamentals_X_input_graph.reset_index(drop=True)

        plt.scatter(Fundamentals_X_input_graph_reset["Avg Daily %Return"],\
                    Fundamentals_X_input_graph_reset["95% VaR"])
        plt.annotate(x,(Fundamentals_X_input_graph_reset["Avg Daily %Return"],Fundamentals_X_input_graph_reset["95% VaR"]),\
                     textcoords="offset points", xytext=(7,2), ha='center')
        plt.xlabel("Avg Daily %Return - seek max daily %Return")
        plt.ylabel("95% VaR - higher VaR = lower max Daily Drawdown, seek max 95% VaR")

    plt.figure(figsize=(18,11))
    plt.grid()
    plt.title("MA to TP %(X) vs 30day Vol(Y) - Relationship between Disc/Prem of 50D MA & 30D Vol")
    for x in Fundamentals_X_input["Tickers"]:
        Fundamentals_X_input_graph = Fundamentals_X_input.loc[lambda Fundamentals_X_input: Fundamentals_X_input['Tickers']== x, :]
        Fundamentals_X_input_graph_reset = Fundamentals_X_input_graph.reset_index(drop=True)

        plt.scatter(Fundamentals_X_input_graph_reset["MA to TP %"],\
                    Fundamentals_X_input_graph_reset["30day Vol"])
        plt.annotate(x,(Fundamentals_X_input_graph_reset["MA to TP %"],Fundamentals_X_input_graph_reset["30day Vol"]),\
                     textcoords="offset points", xytext=(7,2), ha='center')
        plt.xlabel("MA to TP % - higher MAtTP% = lower current price to average price, seek max MAtTP%")
        plt.ylabel("30 day Vol - lower Vol = lower unsystematic risk, seek min Vol")

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
