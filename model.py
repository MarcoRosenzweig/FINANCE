import utils
import numpy as np
from scipy.signal import argrelextrema

class MODEL():
    '''
    Simple model to trade stocks, cryptos etc.
    Initialization:
        - data: historical data of asset prices.
        - buy_locs, sell_locs: locations for buy and sell signals.
    '''
    def __init__(self, tickers, data=None, buy_delay=1):
        if not any([isinstance(tickers, list), isinstance(tickers, str)]):
            raise TypeError('[ERROR]: Input of "tickers" must either be str or list.')
        if isinstance(tickers, str):
            self.tickers = [tickers]
        self.data = data
        self.local_min, self.local_max, self.grad = None, None, None
        self.buy_delay = buy_delay

    def get_data(self, value='Close', filter_date_range=None, *args, **kwargs):
        self.data = utils.download_data(tickers=self.tickers, \
                                        value=value, \
                                        *args, **kwargs)

        if filter_date_range is not None:
            self.apply_date_filter(filter_date_range=filter_date_range)

    def apply_date_filter(self, filter_date_range):
        try:
            self.data = self.data.reindex(filter_date_range)
            self._print_issue('INFO', 'filter applied.')
        except KeyError:
            self._print_issue('WARNING', 'filter not in data.')

###############################################################################
#   INTERNAL FUNCTIONS
###############################################################################
    def _calc_ema(self, data, average_sample):
        '''
        Function to calculate the exponential moving average
        Inputs:
            - average_sample: number of days to average
        Outputs:
            - pandas.core.series.Series, i.e. ema prices.
        '''
        return data.ewm(span=average_sample, adjust=False).mean()

    def _init_model(self, periods=(12, 26, 9)):
        '''
        Function to set up the price model. The idea is to locate the inflection
        points of the difference of "moving average converging diverging (macd)"
        and "Signal Line (signal_line)". These indicate local up and down trends.
        The actual buy and sell prices are therefore the next day, i.e. buy_delay.
        Inputs:
            - values_of_interest: days to calculate the macd (first two values)
            and Signal Line (last value).
                default: 12, 26, 9
            - buy_delay: buy and sell dates
                default: 1
            - grad_return: return the "gradient" of the model, i.e. the model itself
                default: True
        Outputs:
            - local_min: Buy prices
            - local_max: Sell prices
            - grad: "gradient" of the model (optionally)
        '''
        macd = self._calc_ema(self.data, periods[0]) - \
               self._calc_ema(self.data, periods[1])
        signal_line = self._calc_ema(macd, periods[2])
        grad = np.gradient(macd - signal_line)
        grad_dict = dict.fromkeys(self.tickers)
        if isinstance(grad, list):
            local_min, local_max = {}, {}
            self._print_issue('WARNING', 'Ignoring second entry of gradient!')
            grad = grad[0].T
            for n in range(grad.shape[0]):
                local_min[self.tickers[n]] = argrelextrema(grad[n], np.less)
                local_max[self.tickers[n]] = argrelextrema(grad[n], np.greater)
        else:
            local_min = argrelextrema(grad, np.less)[0]
            local_max = argrelextrema(grad, np.greater)[0]
        #transforming grad as dict
        print(grad.size)
        for n, ticker in enumerate(self.tickers):
            grad_dict[ticker] = grad[n]

#        try:
            #if local_max[0] < local_min[0]:
        #        local_max = local_max[1:]
        #except ValueError:
    #        pass
        #this will be handled later!
            #case where one has one sell data without a buy data
        self.local_min = local_min
        self.local_max = local_max
        self.grad = grad_dict

###############################################################################
#   USEFUL FUNCTIONS
###############################################################################
    def _print_issue(self, key, issue, do_print=True):
        if do_print:
            print('[{}]: {}'.format(key, issue))
