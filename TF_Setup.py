import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_FOLDER_TF = "Tensorflow_DataFrame_csv"
try:
    os.mkdir(OUTPUT_FOLDER_TF)
except FileExistsError:
    pass

def tf_setup(OUTPUT_FOLDER_CPSP, output_folder_tf=OUTPUT_FOLDER_TF, export_table=True):
    #Join all Option_Analysis_CPSP CSVs to create one large DF for Tensorflow
    if os.path.isfile('{}/2020_09_11/Option_Analysis_CPSP_2020_09_11.csv'.format(OUTPUT_FOLDER_CPSP)) and \
    os.path.isfile('{}/2020_09_18/Option_Analysis_CPSP_2020_09_18.csv'.format(OUTPUT_FOLDER_CPSP)):
        valid_200911_df = pd.read_csv('{}/2020_09_11/Option_Analysis_CPSP_2020_09_11.csv'\
                                              .format(OUTPUT_FOLDER_CPSP), index_col=0)
        valid_200918_df = pd.read_csv('{}/2020_09_18/Option_Analysis_CPSP_2020_09_18.csv'\
                                              .format(OUTPUT_FOLDER_CPSP), index_col=0)

        tf_df_setup = valid_200911_df.append(valid_200918_df, ignore_index=True)

    #folder_suffix shows date at which Tensorflow_DataFrame_csv_ was last updated
    today = datetime.date.today()
    folder_suffix = str(today).replace("-", "_")
    try:
        folder = os.path.join(output_folder_tf, folder_suffix)
        os.mkdir(folder)
    except FileExistsError:
        pass

    table_name = os.path.join(folder, "Tensorflow_DataFrame_csv_{}.csv".format(folder_suffix))
    if export_table:
        tf_df_setup.to_csv(table_name)
        print("Exported: {}".format(table_name))

    return tf_df_setup
