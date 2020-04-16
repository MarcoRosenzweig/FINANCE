import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import utils
import numpy as np

def plot_model(model, tickers='all', plot_range=None, savefig=None):
    '''
    Function to plot a model.
    Inputs:
        - model: model of class MODEL
        - start:
        - grad: "gradient" of price model
        - locs: "locations" of price model
    '''
    if tickers=='all':
        tickers = model.tickers
    else:
        tickers = utils.check_ticker_input(tickers_input=tickers, \
                                           tickers_avail=model.tickers, \
                                           do_print=True)
    for ticker in tickers:
        if plot_range is None:
            x_axis = model.data[ticker].index
            indices = np.arange(0, x_axis.shape[0], 1)
        else:
            x_axis = model.data[ticker][plot_range].index
            indices = np.where(np.isin(model.data[ticker].index, plot_range))[0]

        grad = model.grad[ticker][indices]
        min_arg = np.where(model.local_min[ticker] >= indices[0])
        max_arg = np.where(model.local_max[ticker] >= indices[0])
        local_min = model.local_min[ticker][min_arg]
        local_max = model.local_max[ticker][max_arg]

        plt.figure(figsize=(16, 9))
        ax1 = plt.subplot(2, 1, 1)
        plt.fill_between(x_axis, 0, grad, \
                         where=grad > 0, \
                         facecolor='green', interpolate=True, label='Up Trend')
        plt.fill_between(x_axis, 0, grad, \
                         where=grad <= 0, \
                         facecolor='red', interpolate=True, label='Down Trend')
        plt.vlines(model.data[ticker].index[local_min], \
                   np.min(grad), np.max(grad), \
                   color='g', label='Min Reached')
        plt.vlines(model.data[ticker].index[local_max], \
                   np.min(grad), np.max(grad), \
                   color='r', label='Peak Reached')
        plt.legend()
        plt.grid()
        #subplot 2:
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        price = model.data[ticker][indices]
        try:
            buy_dates = model.ticker_df[ticker]['Buy Dates'].values[max_arg[0]]
            sell_dates = model.ticker_df[ticker]['Sell Dates'].values[max_arg[0]]
        except IndexError:
            buy_dates = model.ticker_df[ticker]['Buy Dates'].values[max_arg[0][:-1]]
            sell_dates = model.ticker_df[ticker]['Sell Dates'].values[max_arg[0][:-1]]
        plt.plot(x_axis, price, \
                 label='{} price'.format(ticker))
        plt.vlines(buy_dates, np.min(price), np.max(price), \
                   color='g', label='Buy Dates')
        plt.vlines(sell_dates, \
                   np.min(price), np.max(price), \
                   color='r', linestyle='--', label='Sell dates')
        plt.legend()
        plt.grid()

        return ax1, ax2
        #plt.show()
