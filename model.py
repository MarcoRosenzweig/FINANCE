import utils

class MODEL():
    '''
    Simple model to trade stocks, cryptos etc.
    Initialization:
        - data: historical data of asset prices.
        - buy_locs, sell_locs: locations for buy and sell signals.
    '''
    def __init__(self, tickers, data=None):
        if not any([isinstance(tickers, list), isinstance(tickers, str)]):
            raise TypeError('[ERROR]: Input of "tickers" must either be str or list.')
        self.tickers = tickers
        self.data = data
        self.buy_locs, self.sell_locs = None, None

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
    def _calc_EMA(average_sample):
        '''
        Function to calculate the exponential moving average
        Inputs:
            - average_sample: number of days to average
        Outputs:
            - pandas.core.series.Series, i.e. ema prices.
        '''
        return self.data.ewm(span=average_sample, adjust=False).mean()
###############################################################################
#   USEFUL FUNCTIONS
###############################################################################
    def _print_issue(self, key, issue, do_print=True):
        if do_print:
            print('[{}]: {}'.format(key, issue))
