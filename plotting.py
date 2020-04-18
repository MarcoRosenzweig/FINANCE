import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import utils
import numpy as np

def plot_model(model, tickers='all', plot_range=None, plot_from_index=None, \
               plot_from_date=None, plot_break_values=True, \
               *args, **kwargs):

    '''
    Function to plot a model.
    Inputs:
        - model: model of class MODEL
        - tickers: tickers to plot
            default: all, i.e. tickers in input class MODEL
        - plot_range: range to plot of type pandas.date_range()
            defualt: None, i.e. complete data set
        - plot_break_values: if available, plot break_values of input class MODEL
            default: True
    '''
    if tickers=='all':
        tickers = model.tickers
    else:
        tickers = utils.check_ticker_input(tickers_input=tickers, \
                                           tickers_avail=model.tickers, \
                                           do_print=True)
    for ticker in tickers:
        if plot_range is not None:
            x_axis = model.data[ticker][plot_range].index
            indices = np.where(np.isin(model.data[ticker].index, plot_range))[0]
        elif plot_from_index is not None:
            x_axis = model.data[ticker].index[plot_from_index:]
            indices = np.arange(plot_from_index, \
                                model.data[ticker].index.shape[0], 1)
        elif plot_from_date is not None:
            idx = model.data[ticker].index.get_loc(plot_from_date).start
            x_axis = model.data[ticker].index[idx:]
            indices = np.arange(idx, model.data[ticker].index.shape[0], 1)
        else:
            x_axis = model.data[ticker].index
            indices = np.arange(0, x_axis.shape[0], 1)

        grad = model.grad[ticker][indices]
        min_arg = np.where(model.local_min[ticker] >= indices[0])
        max_arg = np.where(model.local_max[ticker] >= indices[0])
        try:
            local_min = model.local_min[ticker][min_arg]
            local_max = model.local_max[ticker][max_arg]
            in_loop = False
        except TypeError:
            #loop over tickers:
            in_loop = True
            local_min = model.local_min[ticker][0][min_arg[1]]
            local_max = model.local_max[ticker][0][max_arg[1]]

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
        try:
            plt.title(ticker, kwargs['title'])
        except KeyError:
            plt.title(ticker, fontsize=14)
        plt.legend(loc='upper left')
        plt.grid()
        #subplot 2:
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        price = model.data[ticker][indices]
        try:
            buy_dates = model.ticker_df[ticker]['Buy Dates'].values[min_arg[0]]
            if in_loop:
                buy_dates = model.ticker_df[ticker]['Buy Dates'].values[min_arg[1]]
        except IndexError:
            utils._print_issue('INFO', 'New buy signal was detected for last value: {}.'.format(model.data[ticker][-1]))
            buy_dates = model.ticker_df[ticker]['Buy Dates'].values[min_arg[0][:-1]]
            if in_loop:
                buy_dates = model.ticker_df[ticker]['Buy Dates'].values[min_arg[1][:-1]]
            buy_dates = np.hstack((buy_dates, model.data[ticker].index[local_min[-1] + 1].to_numpy()))
        try:
            sell_dates = model.ticker_df[ticker]['Sell Dates'].values[max_arg[0]]
            if in_loop:
                sell_dates = model.ticker_df[ticker]['Sell Dates'].values[max_arg[1]]
        except IndexError:
            utils._print_issue('INFO', 'New sell signal was detected for last value: {}.'.format(model.data[ticker][-1]))
            sell_dates = model.ticker_df[ticker]['Sell Dates'].values[max_arg[0][:-1]]
            if in_loop:
                sell_dates = model.ticker_df[ticker]['Sell Dates'].values[max_arg[1][:-1]]
            sell_dates = np.hstack((sell_dates, model.data[ticker].index[local_max[-1] + 1].to_numpy()))

        plt.plot(x_axis, price, \
                 label='{} price'.format(ticker))
        plt.vlines(buy_dates, np.min(price), np.max(price), \
                   color='g', label='Buy Dates')
        plt.vlines(sell_dates, \
                   np.min(price), np.max(price), \
                   color='r', linestyle='--', label='Sell dates')
        if plot_break_values:
            if model.break_values is not None:
                plt.hlines(model.break_values[ticker][0], x_axis[0], x_axis[-1], \
                           color='k', label='Break value w.r.t today')
                plt.hlines(model.break_values[ticker][1], x_axis[0], x_axis[-1], \
                           color='c', label='Break value w.r.t yesterday')

        plt.legend(loc='upper left')
        plt.grid()
        #return ax1, ax2
        #plt.show()
