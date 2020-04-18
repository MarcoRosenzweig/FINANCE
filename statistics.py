import utils
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_probs(model, time=None, tickers='all', stats_data=None, \
               auto_update_tolerances=False, *args, **kwargs):

    if tickers == 'all':
        tickers = model.tickers
    else:
        tickers = utils.check_ticker_input(tickers_input=tickers, \
                                           tickers_avail=model.tickers)
    try:
        timezone = kwargs['timezone']
    except KeyError:
        timezone = None
    try:
        start = kwargs['start']
    except KeyError:
        start = None
    for ticker in tickers:
        utils._print_issue(None, '=' * 80)
        utils._print_issue('INFO', 'Current ticker: {}'.format(ticker))
        z_values = _create_z_values(model=model, ticker=ticker, \
                                    stats_data=stats_data, timezone=timezone, \
                                    start=start, \
                                    auto_update_tolerances=auto_update_tolerances)

        freq_range, frequencies = _create_freq()
        delta_t = model.data.index[-1].to_datetime64() - pd.Timestamp.now().to_datetime64()
        delta_t = pd.Timedelta(delta_t).seconds / 3600
        #plots:
        fig = plt.figure(figsize=(16, 9))
        ax1 = plt.subplot(211)
        probs = (1 - ss.norm.cdf(z_values)) * 100
        ax1.plot(frequencies, probs[0], label='Probability for smallest tolerance.')
        plt.vlines(delta_t, np.min(probs), np.max(probs), label='Time to deadline.')
        deg = 5
        poly_line = np.poly1d(np.polyfit(freq_range, probs[0], deg))
        ax1.plot(frequencies, poly_line(freq_range), 'r', label='Polyfit of deg {}'.format(deg))
        ax1.invert_xaxis()
        ax1.grid()
        ax1.legend()
        prob_small = poly_line(delta_t)
        print('Probability for reaching smallest tolerance: {}%'.format(prob_small))
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(frequencies, probs[1], label='Probability for highest tolerance.')
        plt.vlines(delta_t, np.min(probs), np.max(probs), label='Time to deadline.')
        poly_line = np.poly1d(np.polyfit(freq_range, probs[1], deg))
        ax2.plot(frequencies, poly_line(freq_range), 'r', label='Polyfit of deg {}'.format(deg))
        ax2.grid()
        ax2.legend()
        prob_high = poly_line(delta_t)
        print('Probability for reaching highest tolerance: {}%'.format(prob_high))
        print('Probability between: {}%'.format(np.abs(prob_high - prob_small)))

def _create_z_values(model, ticker, stats_data=None, \
                     auto_update_tolerances=False, *args, **kwargs):
    freq_range, frequencies = _create_freq()
    try:
        timezone = kwargs['timezone']
    except KeyError:
        timezone = None
    try:
        start = kwargs['start']
    except KeyError:
        start = None
    _, means, stds = _get_price_moves_and_stats(ticker=ticker, \
                                                stats_data=stats_data,
                                                timezone=timezone, \
                                                start=start)
    if auto_update_tolerances:
        utils._print_issue('INFO', 'Auto update of tolerances!')
        current_value = utils.download_data(tickers=ticker, \
                                            start=pd.Timestamp.today(), \
                                            value='Close').values[-1]
        current_tols = model.break_values[ticker] - current_value
        utils._print_issue('INFO', 'Current value: {}!'.format(current_value))
        utils._print_issue('INFO', 'New tolerances: {}!'.format(current_tols))
        tol_unten = np.sort(current_tols)[0]
        tol_oben = np.sort(current_tols)[1]
    else:
        tol_unten = np.sort(model.tolerances[ticker])[0]
        tol_oben = np.sort(model.tolerances[ticker])[1]
    z_values_unten = (tol_unten - means) / stds
    z_values_oben = (tol_oben - means) / stds
    return np.array([z_values_unten, z_values_oben])

def _get_price_moves_and_stats(ticker, stats_data=None, \
                               timezone=None, start=None):
    if timezone is None:
        timezone = 'Europe/London'
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
                                    freq=freq, tz=timezone, name='Datetime')
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
