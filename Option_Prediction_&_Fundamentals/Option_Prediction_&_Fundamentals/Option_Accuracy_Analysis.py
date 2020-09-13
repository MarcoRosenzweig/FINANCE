import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display
from option_prediction import OUTPUT_FOLDER

OUTPUT_FOLDER_CPSP = "Option_Analysis_CPSP_csv"
try:
    os.mkdir(OUTPUT_FOLDER_CPSP)
except FileExistsError:
    pass

def actual_prices(date_of_interest, days_to_go_back, Company_Names, OUTPUT_FOLDER,\
                  output_folder_cpsp=OUTPUT_FOLDER_CPSP, export_table=True):

    '''Function to determine actual price at expiry and comparing it to predicted price (incl. direction)'''
    #1.create dates as list with the same format as folder names:

    full_prediction_df = pd.DataFrame({})
    dates = pd.date_range(end=datetime.date.today(), periods=days_to_go_back, freq='d').strftime('%Y_%m_%d').to_list()
    #get valid dataframes:
    for date in dates:
        if os.path.isfile('{0}/{1}/Option_Weekly_{1}.csv'.format(OUTPUT_FOLDER, date)) and \
        os.path.isfile('{0}/{1}/Company_Fundamentals_{1}.csv'.format(OUTPUT_FOLDER, date)):
            valid_op_dataFrames = pd.read_csv('{0}/{1}/Option_Weekly_{1}.csv'.format(OUTPUT_FOLDER, date), index_col=0)

            valid_fdmtls_dataFrames = pd.read_csv('{0}/{1}/Company_Fundamentals_{1}.csv'.format(OUTPUT_FOLDER, date),\
                                                  index_col=0)
            df_outer = [pd.merge(valid_op_dataFrames, valid_fdmtls_dataFrames, on='Tickers', how='outer')]
            for x in df_outer:

                full_prediction_df = full_prediction_df.append(x, ignore_index=True)

        else:
            print("0")

        #df_outer = pd.merge(valid_op_dataFrames, valid_fdmtls_dataFrames, on='Tickers', how='outer')
        #df_outer_adj = df_outer.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)
    #pd.set_option("display.max_rows", None, "display.max_columns", None)


    Option_Analysis_df = full_prediction_df.loc[full_prediction_df['Option Exp Date'] == date_of_interest]
    #dataFrames_to_merge = [df.loc[df['Tickers'] != 'AAPL'].loc[df['Tickers'] != 'TSLA'] for df in dataFrames_to_merge]
    #Option_Analysis_df = pd.concat(dataFrames_to_merge)
    Option_Analysis_df.sort_values(by=['Tickers', 'Days_to_Exp'], inplace=True)
    #display(Option_Analysis_df)

    #3. Add actual prices at Expiry for the date the csv tried to predict
    Option_Analysis_loadin = Option_Analysis_df.loc[lambda Option_Analysis_df: \
                                                    Option_Analysis_df['Option Exp Date']== date_of_interest, :]
    Option_Analysis_loadin_reset = Option_Analysis_loadin.reset_index(drop=True)
    Option_Analysis_loadin_tickers = list(Option_Analysis_loadin_reset["Tickers"])

    ticker_t_p_list = []

    for x in Option_Analysis_loadin_tickers:
        #To get Actual Price Data of Ticker list
        tickerp = yf.download(x, start= date_of_interest)
        ticker_p = tickerp['Close']
        ticker_t_p = ticker_p[1]
        ticker_t_p_list.append(ticker_t_p)

    #Add Prices at Exp to DF
    Option_Analysis_loadin_reset["Actual Price @ Expiry"] = ticker_t_p_list
    Option_Analysis_Final = Option_Analysis_loadin_reset

    #4. Clarify Cols and calculate Expected - Actual change
    Option_Analysis_Final["Price @ DtE"] = Option_Analysis_Final["Todays Price"]
    Option_Analysis_Final["Expected %Change"] = (Option_Analysis_Final["CPSP"]-Option_Analysis_Final["Price @ DtE"])/\
    Option_Analysis_Final["Price @ DtE"]*100

    Option_Analysis_Final["Abs_AP_to_CPSP_Diff"] = Option_Analysis_Final["CPSP"]- \
    Option_Analysis_Final["Actual Price @ Expiry"]
    Option_Analysis_Final["%_AP_to_CPSP_Diff"] = (Option_Analysis_Final["Abs_AP_to_CPSP_Diff"] / \
                                                      Option_Analysis_Final["Actual Price @ Expiry"])*100
    Option_Analysis_Final["Prediction_Accuracy_%_CPSP"] = 100 + Option_Analysis_Final["%_AP_to_CPSP_Diff"]
    Option_Analysis_Final["Actual %Change"] = (Option_Analysis_Final["Actual Price @ Expiry"] -\
    Option_Analysis_Final["Price @ DtE"])/Option_Analysis_Final["Price @ DtE"]*100

    Option_Analysis_Final["Expected - Actual"] = Option_Analysis_Final["Expected %Change"] - \
    Option_Analysis_Final["Actual %Change"]

    Option_Analysis_CPSP = Option_Analysis_Final[["Tickers","Days_to_Exp","Price @ DtE","CPSP",\
                                                 "Actual Price @ Expiry",\
                                                 "Expected %Change", "Actual %Change","Expected - Actual",\
                                                 "MarketCap","Beta","50 Day MA","MA to TP %","Sharpe Ratio",\
                                                 "30day Vol", "95% VaR"]]
    emin = min(Option_Analysis_CPSP["Expected %Change"])
    amin = min(Option_Analysis_CPSP["Actual %Change"])
    mins = [emin, amin]

    emax = max(Option_Analysis_CPSP["Expected %Change"])
    amax = max(Option_Analysis_CPSP["Actual %Change"])
    maxs = [emax, amax]

    min_axis = min(mins)
    max_axis = max(maxs)

    min_max_axis = [min_axis,max_axis]
    axiss_final = max(np.abs(min_max_axis))
    minaxis = -axiss_final
    maxaxis = axiss_final

    #5. Plot accross tickers and DtE
    plt.figure(figsize=(12,12))

    for x in Company_Names:
        Option_Analysis_CPSPticker = Option_Analysis_CPSP.loc[lambda Option_Analysis_CPSP: \
                                                        Option_Analysis_CPSP['Tickers']== x, :]

        Option_Analysis_CPSPticker_reset = Option_Analysis_CPSPticker.reset_index(drop=True)

        #Option_Analysis_CPSPticker_reset["Expected %Change Adj"] = \
        #Option_Analysis_CPSPticker_reset["Expected %Change"] \
        #- DTE0_m

        plt.scatter(Option_Analysis_CPSPticker_reset["Expected %Change"],\
                    Option_Analysis_CPSPticker_reset["Actual %Change"], label = x)

        plt.gca().set_xlim(minaxis-1,\
                           maxaxis+1)
        plt.gca().set_ylim(minaxis-1,\
                           maxaxis+1)
    plt.grid()
    plt.legend()
    #print(DTE0_m)

    plt.figure(figsize=(12,12))

    for x in Option_Analysis_CPSPticker_reset["Days_to_Exp"]:
        Option_Analysis_CPSP_days = Option_Analysis_CPSP.loc[lambda Option_Analysis_CPSP: \
                                                    Option_Analysis_CPSP['Days_to_Exp']== x, :]
        Option_Analysis_CPSP_days_reset = Option_Analysis_CPSP_days.reset_index(drop=True)

        DTE0_m = Option_Analysis_CPSP["Expected - Actual"].mean()

        #Option_Analysis_CPSP_days_reset["Expected %Change Adj"] = \
        #Option_Analysis_CPSP_days_reset["Expected %Change"] - \
        #DTE0_m

        plt.scatter(Option_Analysis_CPSP_days_reset["Expected %Change"],\
                    Option_Analysis_CPSP_days_reset["Actual %Change"], label = x)

        plt.gca().set_xlim(minaxis-1,\
                           maxaxis+1)
        plt.gca().set_ylim(minaxis-1,\
                           maxaxis+1)
    plt.grid()
    plt.legend()

    #6. Analyse how many predictions were made in the correct direction
    ePredictions = []
    aPredictions = []

    for x in Option_Analysis_CPSP["Expected %Change"]:
        if x > 0:
            ePredictions.append("positive")
        else:
            ePredictions.append("negative")

    for x in Option_Analysis_CPSP["Actual %Change"]:
        if x > 0:
            aPredictions.append("positive")
        else:
            aPredictions.append("negative")

    Prediction = []
    for x,y in zip(ePredictions, aPredictions):
        if x == y:
            Prediction.append("Correct")
        else:
            Prediction.append("False")

    da = Prediction.count("Correct")/np.shape(Option_Analysis_CPSP)[0]*100

    shape = np.shape(Option_Analysis_CPSP)[0]

    print(Prediction.count("Correct"), "correctly predicted out of",shape," hence a ",da,"% Accuracy" )
    print(Prediction.count("False"), "incorrectly predicted out of", shape)
    Option_Analysis_CPSP["predicted Direction"] = Prediction

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    Option_Analysis_CPSP[["Tickers","Days_to_Exp","Price @ DtE","CPSP", "Actual Price @ Expiry",\
                          "Expected %Change","Actual %Change","predicted Direction","Expected - Actual",\
                          "MarketCap","Beta","50 Day MA","MA to TP %","Sharpe Ratio",\
                          "30day Vol", "95% VaR"]]
    folder_suffix = date_of_interest.replace("-", "_")
    try:
        folder = os.path.join(output_folder_cpsp, folder_suffix)
        os.mkdir(folder)
    except FileExistsError:
        pass

    table_name = os.path.join(folder, "Option_Analysis_CPSP_{}.csv".format(folder_suffix))
    if export_table:
        Option_Analysis_CPSP.to_csv(table_name)
        print("Exported: {}".format(table_name))

    return Option_Analysis_CPSP
