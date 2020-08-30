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

def create_option_dates(n_option_dates):
    today = datetime.date.today()
    fridays = today + datetime.timedelta((4 - today.weekday()) % 7)
    return pd.date_range(periods=n_option_dates,
                         start=fridays,
                         freq='7d').strftime("%Y-%m-%d").to_list()

def option_prediction (Company_Names, option_dates, output_folder=OUTPUT_FOLDER, export_table=True):
    """
    Inputs:
        - Company_Names as ticker list
        - option_dates as date list in format %Y-%M-D
    Returns:
        - DataFrame

    """
    #Loop to calculate Weighted Average Strike Price (predicted Price) between Put & Call Options
    ticker_t_p_list = []
    price_change_list = []
    Calls_WOI_list = []
    Puts_WOI_list = []
    CP_WMid_list = []

    for x in Company_Names:
        #To get Options Data of Ticker list
        ticker = yf.Ticker(x)

        #To get Price Data of Ticker list
        tickerp = yf.download(x, start= datetime.date.today() - datetime.timedelta(days=2))
        ticker_p = tickerp['Close']
        ticker_t_p = ticker_p[-1]
        ticker_t_p_list.append(ticker_t_p)
        ticker_ytd_p = ticker_p[-2]
        price_change = (ticker_t_p -ticker_ytd_p)/ ticker_ytd_p *100
        price_change = round(price_change,4)
        price_change_list.append(price_change)

        for y in option_dates:
            #To adjust the Options Dataframe for calls and puts
            tickeroptioninfo = ticker.option_chain(y)
            calls_strikes_OI = tickeroptioninfo.calls[["contractSymbol","strike","openInterest"]]
            calls_strikes_OI = calls_strikes_OI.replace(0, np.nan)
            calls_strikes_OI_dropna = calls_strikes_OI.dropna(how='any', axis=0)
            calls_strikes_OI_dropna_droph = calls_strikes_OI_dropna.drop(calls_strikes_OI_dropna['openInterest'].idxmax())
            calls_strikes_OI = calls_strikes_OI_dropna_droph["strike"]
            calls_openInterest = calls_strikes_OI_dropna_droph["openInterest"]

            puts_strikes_OI = tickeroptioninfo.puts[["strike","openInterest"]]
            puts_strikes_OI = puts_strikes_OI.replace(0, np.nan)
            puts_strikes_OI_dropna = puts_strikes_OI.dropna(how='any', axis=0)
            puts_strikes_OI_dropna_droph = puts_strikes_OI_dropna.drop(puts_strikes_OI_dropna['openInterest'].idxmax())
            puts_strikes_OI = puts_strikes_OI_dropna_droph["strike"]
            puts_openInterest = puts_strikes_OI_dropna_droph["openInterest"]

            #Throwing out the tails of the Open Interest distribution to not get a distorted calculation
            #(basically getting rid of the largest OI and all those that are smaller than 5% of the largest)
            Top_Call_OI = calls_strikes_OI_dropna_droph['openInterest'].max()
            Top_Put_OI = puts_strikes_OI_dropna_droph['openInterest'].max()
            Top_CP_OI = [Top_Call_OI, Top_Put_OI]
            Top_Limit_OI = max(Top_CP_OI)

            limit = (Top_Limit_OI / 20)
            bigP_OI = puts_strikes_OI_dropna_droph[puts_strikes_OI_dropna_droph['openInterest'] > limit]
            bigC_OI = calls_strikes_OI_dropna_droph[calls_strikes_OI_dropna_droph['openInterest'] > limit]

            mylist = []

            for c_n in Company_Names:
                df = (c_n,) * np.shape(option_dates)[0]
                mylist.append(df)

            C_N_df = list(itertools.chain(*mylist))

            option_dates_df = option_dates * np.shape(Company_Names)[0]

            Calls_OI_PCT=[]

            #Calculating the Weighted OI Average for puts and Calls
            for z in bigC_OI['openInterest']:
                oi = z / np.sum(bigC_OI['openInterest'])
                Calls_OI_PCT.append(oi)

            bigC_OI['OI_PCT'] = Calls_OI_PCT
            bigC_WOI = bigC_OI[['strike','openInterest', 'OI_PCT']]

            bigC_WOI['WOI'] = bigC_WOI['strike'] * bigC_WOI['OI_PCT']
            Calls_WOI = np.sum(bigC_WOI['WOI'])
            Calls_WOI_list.append(Calls_WOI)

            Puts_OI_PCT=[]

            for a in bigP_OI['openInterest']:
                oi = a / np.sum(bigP_OI['openInterest'])
                Puts_OI_PCT.append(oi)

            bigP_OI['OI_PCT'] = Puts_OI_PCT
            bigP_WOI = bigP_OI[['strike','openInterest', 'OI_PCT']]

            bigP_WOI['WOI'] = bigP_WOI['strike'] * bigP_WOI['OI_PCT']
            Puts_WOI = np.sum(bigP_WOI['WOI'])
            Puts_WOI_list.append(Puts_WOI)

            #Calculating the Weighted Midpoint and implied difference to today
            Sum_OI = np.shape(bigP_WOI)[0] + np.shape(bigC_WOI)[0]
            CP_WMid = (Calls_WOI*(np.shape(bigC_WOI)[0]/Sum_OI) + Puts_WOI*(np.shape(bigP_WOI)[0]/Sum_OI))
            CP_WMid_list.append(CP_WMid)

            for b in ticker_t_p_list:
                CP_WMid_Price_Diff = CP_WMid - b

                CP_WMid_Price_Diff_Pct = CP_WMid_Price_Diff / b *100
    #Set up DF-shape as function of Companies and Expiry Dates
    myprices = []

    for x in ticker_t_p_list:
        dfprices = (x,) * np.shape(option_dates)[0]
        myprices.append(dfprices)

    Prices_df = list(itertools.chain(*myprices))

    x_dates = []
    for x in option_dates_df:
        x_date = datetime.datetime.strptime(x, "%Y-%m-%d")
        x_dates.append(x_date)

    Today = [None] * (np.shape(Company_Names)[0]*np.shape(option_dates)[0])
    Today = [datetime.datetime.today() if x==None else x for x in Today]
    Days_to_Exp = [(a - b).days for a, b in zip(x_dates, Today)]
    #Weekly Price Changes from TODAY DF
    Option_Analysis_T0 = pd.DataFrame({'Tickers': C_N_df,
                                       "Todays Price": Prices_df,
                                       'Option Exp Date': option_dates_df,
                                       "Days_to_Exp": Days_to_Exp,
                                       "Calls_WOI": Calls_WOI_list,
                                       "Puts_WOI": Puts_WOI_list,
                                       "Predicted_Opt_Price":CP_WMid_list})
    Option_Analysis_T0["implied Change from T0"] = Option_Analysis_T0["Predicted_Opt_Price"] - Option_Analysis_T0["Todays Price"]
    Option_Analysis_T0["implied %Change from T0"] = (Option_Analysis_T0["implied Change from T0"] / Option_Analysis_T0["Todays Price"])*100
    #Inter-Weekly Changes DF - X-train/test for TF
    Pct_Change_T1_df=[]
    Price_diff_T1_l_df = []

    for x in Company_Names:
        Option_Analysis_T1 = Option_Analysis_T0.loc[lambda Option_Analysis_T0: Option_Analysis_T0['Tickers']== x, :]
        Option_Analysis_T1_reset = Option_Analysis_T1.reset_index(drop=True)
        Price_diff_T1 = np.diff(Option_Analysis_T1_reset["implied Change from T0"])
        Price_diff_T1_l = list(Price_diff_T1)

        Pct_Change_T1 = list(Price_diff_T1_l / Option_Analysis_T1_reset["Predicted_Opt_Price"][:3]*100)
        Pct_Change_initial = Option_Analysis_T1_reset["implied %Change from T0"][0]
        Price_Change_initial = Option_Analysis_T1_reset["implied Change from T0"][0]

        Pct_Change_T1.insert(0,Pct_Change_initial)
        Price_diff_T1_l.insert(0, Price_Change_initial)

        for y in Pct_Change_T1:
            Pct_Change_T1_df.append(y)

        for z in Price_diff_T1_l:
            Price_diff_T1_l_df.append(z)

    Option_Analysis_Weekly = pd.DataFrame({'Tickers': C_N_df,
                                           "Todays Price": Prices_df,
                                           'Option Exp Date': option_dates_df,
                                           "Days_to_Exp": Days_to_Exp,
                                           "Calls_WOI": Calls_WOI_list,
                                           "Puts_WOI": Puts_WOI_list,
                                           "Predicted_Opt_Price": CP_WMid_list,
                                           "implied Change to prior week": Price_diff_T1_l_df,
                                           "implied %Change to prior week": Pct_Change_T1_df})

    todaydate = datetime.date.today()
    today_str = str(todaydate).replace("-", "_")
    try:
        folder = os.path.join(output_folder, today_str)
        os.mkdir(folder)
    except FileExistsError:
        pass

    table_name = os.path.join(folder, "Option_Weekly_{}.csv".format(today_str))
    if export_table:
        Option_Analysis_Weekly.to_csv(table_name)
        print("Exported: {}".format(table_name))
    return Option_Analysis_Weekly
