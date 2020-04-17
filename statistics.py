import utils
import scipy.stats as ss
import numpy as np
import pandas as pd

def comp_z_values(model, tickers='all', stats_data=None):
    if tickers == 'all':
        tickers = model.tickers
    else:
        tickers = utils.check_ticker_input(tickers_input=tickers, \
                                           tickers_avail=model.tickers)

    for ticker in tickers:
        _, means, stds = _get_price_moves_and_stats(ticker=ticker, \
                                                    stats_data=stats_data)

        for tol in model.tolerances[ticker]:
            print(tol)


def _get_price_moves_and_stats(ticker, stats_data=None, start=None):
    if start is None:
        start = pd.Timestamp(2019, 1, 1, 0)
    if stats_data is None:
        stats_data = utils.download_data(tickers=ticker, start=start, \
                                         interval='60m', value='Close')

    freq_range, frequencies = _create_freq()
    price_movements = dict.fromkeys(frequencies)
    means = np.zeros(freq_range.shape)
    stds = np.zeros(freq_range.shape)
    for index, freq in enumerate(frequencies):
        current_time = start
        current_rng = pd.date_range(start=current_time, end=pd.Timestamp.today(), \
                                    freq=freq, tz='Europe/London', name='Datetime')
        current_moves = np.diff(stats_data[current_rng])
        current_moves = current_moves[~np.isnan(current_moves)]
        means[index] = np.mean(current_moves)
        stds[index] = np.std(current_moves)
        price_movements[freq] = current_moves
    return price_movements, means, stds

def _create_freq():
    freq_range = np.arange(1, 25, 1)
    frequencies = [freq + 'h' for freq in freq_range.astype(str)]
    return freq_range, frequencies
