#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import itertools

OUTPUT_FOLDER = "Option_Weekly_csv"
import os
try:
    os.mkdir(OUTPUT_FOLDER)
except FileExistsError:
    pass

def plot_option_prediction(option_prediction, Company_Names, option_dates, output_folder=OUTPUT_FOLDER, export_figure=True):
    #Plot the price changes and mean as implied in the Options Market

    Option_Analysis_T2_mean_plot = []
    today = datetime.date.today()

    for y in option_dates:
        Option_Analysis_T2 = option_prediction.loc[lambda option_prediction:                                                    option_prediction['Option Exp Date']== y, :]
        Option_Analysis_T2_reset = Option_Analysis_T2.reset_index(drop=True)
        Option_Analysis_T2_mean = np.sum(Option_Analysis_T2_reset['implied %Change to prior week']) /         np.shape(Option_Analysis_T2_reset["implied %Change to prior week"])[0]

        Option_Analysis_T2_mean_plot.append(Option_Analysis_T2_mean)

    fig = plt.figure(figsize=(16,9))
    plt.plot(option_dates, Option_Analysis_T2_mean_plot, c="k", label="Mean", linewidth=3.5)
    plt.xlabel("Dates")
    plt.ylabel("implied weekly % Change")
    plt.title("Price Development according to Weighted Option Averages")

    for x in Company_Names:
        Option_Analysis_T3 = option_prediction.loc[lambda option_prediction:                                                    option_prediction['Tickers']== x, :]
        Option_Analysis_T3_reset = Option_Analysis_T3.reset_index(drop=True)
        plt.plot(option_dates, Option_Analysis_T3_reset["implied %Change to prior week"],                 label=x)
        
    plt.grid()
    plt.legend()

    print(Option_Analysis_T2_mean_plot)
    
    today_str = str(today).replace("-", "_")
    
    try:
        folder = os.path.join(output_folder, today_str)
        os.mkdir(folder)
    except FileExistsError:
        pass

    fig_name = os.path.join(folder, "Option_Weekly_Plot_{}.pdf".format(today_str))
    if export_figure:
        plt.savefig(fig_name)
        print("Exported: {}".format(fig_name))
