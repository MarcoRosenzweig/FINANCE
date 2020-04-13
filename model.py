import utils
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import copy

class MODEL():
    '''
    Simple model to trade stocks, cryptos etc.
    Initialization:
        - data: historical data of asset prices.
        - local_min, local_max, grad: key characterisitics which will be
        set later.
        - buy_delay: delay between detecting signals and actual handling.
            default: 1
        - ticker_df: dictionaries of pandas DataFrame for each ticker.
    '''
    def __init__(self, tickers, data=None, buy_delay=1):
        self.tickers = self._check_ticker_input(tickers=tickers)
        self.data = data
        self.local_min, self.local_max, self.grad = None, None, None
        self.buy_delay = buy_delay
        self.ticker_df = dict.fromkeys(self.tickers)

    def get_data(self, value='Close', filter_date_range=None, *args, **kwargs):
        '''
        Function to get asset historical prices.
        '''
        self.data = utils.download_data(tickers=self.tickers, \
                                        value=value, \
                                        *args, **kwargs)

        if filter_date_range is not None:
            self.apply_date_filter(filter_date_range=filter_date_range)

    def apply_date_filter(self, filter_date_range):
        try:
            filtered_data = self.data.reindex(filter_date_range)
        except KeyError:
            self._print_issue('WARNING', 'filter not in data.')
            return
        else:
            #1 index is ticker, 0 index is data
            nan_values = np.where(np.isnan(filtered_data))[0]
            n_nan_values = nan_values.size
            if  n_nan_values > 0:
                self._print_issue('WARNING', 'Filter would result in {} NaN values.'\
.format(n_nan_values))
                answer = ''
                while answer not in ['y', 'n']:
                    answer = input('[USER-INPUT]: Remove NaN values? [y/n]: ')
                    if answer == 'y':
                        force_filter = True
                    elif answer == 'n':
                        force_filter = False
                    else:
                        self._print_issue('ERROR', 'Answer with "y" or "n".')
                if force_filter:
                    filtered_data = filtered_data.dropna()
            self.data = filtered_data
            self._print_issue('INFO', 'filter applied.')

    def eval_model(self, entry_money=200, fees=(1.0029, .9954), tax=.25, visualize=False, *args, **kwargs):
        '''
        Function to evaluate the price model predictions
        Inputs:
            - data: price data of asset
            - locs: buy and sell locations, i.e. return from from function price_model()
            - entry_money: initial investment
                default = 100
            - fees: fee for buying and selling prices, i.e. buy asset at broker for slightly higher price than actual asset prices, vice versa for sells
                default = (1.005, .995), i.e. .5% higher buy price and .5% lower sell price
            - tax: german tay payments for annual wins > 800€
                default = .25, i.e. 25%
            - df_return: return model evaluation as pandas DataFrame
                default = True
        Outputs:
            - net_income: Net Income/win after entry_money (and possibly tax) subtracted
            - df_return: model evaluation as pandas DataFrame
        '''

        utils.print_opening(ticker=self.tickers, \
                            start_date=self.data.index[0].strftime('%D'), \
                            end_date=self.data.index[-1].strftime('%D'), \
                            initial_investment_per_ticker=entry_money, \
                            do_print=True)

        if any([self.local_min is None, self.local_max is None, self.grad is None]):
            self._init_model()

        for ticker in self.tickers:
            self._print_issue('TICKER', ticker)

            buy_locs, sell_locs = self._get_locs(ticker=ticker)

            buy_prices = self.data[ticker][buy_locs]
            buy_dates = self.data[ticker].index.values[buy_locs]
            sell_prices = self.data[ticker][sell_locs]
            sell_dates = self.data[ticker].index.values[sell_locs]

            buy_prices *= fees[0]
            sell_prices *= fees[1]
            #check if nan in prices:
            #TODO:
            '''
            nan_indices = np.isnan(sell_prices)
            sell_prices = sell_prices[~nan_indices]
            buy_prices = buy_prices[~nan_indices]
            nan_indices = np.isnan(buy_prices)
            sell_prices = sell_prices[~nan_indices]
            buy_prices = buy_prices[~nan_indices]
            '''
            n_calls = sell_prices.shape[0]
            if buy_prices.shape > sell_prices.shape:
                #must use to_numpy() since the dates are still stored in prices as names
                #-> pandas devides same dates, obviously buy and sell dates differ,
                #hence pandas would return NaN all the time
                ratios = sell_prices.to_numpy() / buy_prices.to_numpy()[:-1]
            else:
                ratios = sell_prices.to_numpy() / buy_prices.to_numpy()
            trade_rewards = entry_money * np.cumprod(ratios)
            #Calculate trade wins
            trade_wins = np.diff(trade_rewards)
            #Insert first win
            try:
                trade_wins = np.insert(trade_wins, 0, trade_rewards[0] - entry_money)
            except IndexError:
                #case where one has one buy but not yet selled.
                pass
            #Evaluate Calls
            good_calls = np.where(trade_wins > 0)
            bad_calls = np.where(trade_wins < 0)
            try:
                efficiency = good_calls[0].shape[0] / n_calls
            except ZeroDivisionError:
                efficiency = np.nan
            #TODO: Error handling here:
            win_loss = trade_wins / (trade_rewards - trade_wins)
            average_win = np.mean(win_loss[np.where(win_loss > 0)])
            average_loss = np.mean(win_loss[np.where(win_loss < 0)])
            if np.sum(trade_wins) > 800:
                tax_pays = np.sum(trade_wins) * tax
                self._print_issue('INFO', '{:.2f} tax was paid.'.format(tax_pays))
                net_income = (trade_rewards[-1] - entry_money) * (1 - tax)
            else:
                self._print_issue('INFO', 'No tax paid.')
                net_income = np.sum(trade_wins)
            #create final DataFrame
            #be aware that buy_dates can be 1 entry longer then sell dates!
            if buy_dates.shape[0] > sell_dates.shape[0]:
                if sell_dates.shape[0] > 0:
                    self._print_issue('INFO', 'Last entry of "Sell Dates" will \
be assigned equally as the \
penultimate one.')
                    sell_dates = np.append(sell_dates, sell_dates[-1])
                else:
                    self._print_issue('INFO', 'First entry of "Sell Dates" \
                                       will be first entry of "Buy Dates".')
                    sell_dates = buy_dates[0]
                try:
                    sell_prices.loc[pd.Timestamp.max] = np.nan
                except: #OverflowError: --> NOT WORKING?
                    sell_prices.loc[buy_prices.index[-1]] = np.nan
                trade_rewards = np.append(trade_rewards, np.nan)
                trade_wins = np.append(trade_wins, np.nan)
                win_loss = np.append(win_loss, np.nan)
            final_df = pd.DataFrame(data = {'Buy Dates': buy_dates, \
                                            'Sell Dates': sell_dates, \
                                            'Buy Prices': buy_prices.to_numpy(), \
                                            'Sell Prices': sell_prices.to_numpy(), \
                                            'Trade Reward': trade_rewards, \
                                            'Trade Win': trade_wins, \
                                            'Trade Efficiency': win_loss})
            self.ticker_df[ticker] = final_df
            self._print_issue(None, '-'*82)
            self._print_issue('SUMMARY', \
                              'Average trade win: {:.10%}'.format(average_win))
            self._print_issue('SUMMARY', \
                              'Average trade loss: {:.10%}'.format(average_loss))
            self._print_issue('SUMMARY', \
                              'Efficiency: {:.2%}'.format(efficiency))
            self._print_issue('SUMMARY', \
                              'NET WIN: {:.2f}'.format(net_income))
            self._print_issue(None, '='*82)

    def copy_model(self):
        return copy.deepcopy(self)

    def append_timedelta(self, timedelta):
        new_entry = self.data.index[-1] + pd.Timedelta(days=timedelta)
        idx = pd.date_range(start=self.data.index[0], end=new_entry)
        new_data = self.data.reindex(idx)
        answer = ''
        while answer not in ['y', 'n']:
            answer = input('[USER-INPUT]: Overwrite existing data? [y/n]: ')
            if answer == 'y':
                self.data = new_data
            elif answer == 'n':
                self._print_issue('INFO', 'No values were appended\
to the original data. Modified DataFrame will be returned.')
                return new_data
            else:
                self._print_issue('ERROR', 'Answer with "y" or "n".')
        self._print_issue('INFO', 'NaN values were append.')

    def comp_break_values(self, tickers='all'):
        if tickers == 'all':
            tickers = self.tickers
        else:
            tickers = self._check_ticker_input(tickers=tickers)
        valid_tickers = []
        for ticker in tickers:
            if ticker not in self.tickers:
                self._print_issue('WARNING', 'Ticker "{}" not in self.tickers'.format(ticker))
            else:
                valid_tickers.append(ticker)
        if len(valid_tickers) == 0:
            self._print_issue('ERROR', 'No input ticker in self.tickers.')
            return
        imag_model = self.copy_model()
        break_values_dict = dict.fromkeys(valid_tickers)
        current_values = dict.fromkeys(valid_tickers, None)
        tolerances = dict.fromkeys(valid_tickers)
        for ticker in valid_tickers:
            break_values = [None, None]
            if np.isnan(self.data[ticker].values[-1]):
                value_index = -2
            else:
                value_index = -1
            current_values[ticker] = self.data[ticker].values[value_index]
            #create range:
            deviation = .4
            start_value = current_values[ticker] * (1 - deviation)
            end_value = current_values[ticker] * (1 + deviation)
            step_size = current_values[ticker] / 5000
            rng = np.arange(start_value, end_value, step_size)
            #start algorithm:
            for value in rng:
                imag_model.data[ticker].values[-1] = value
                imag_model._init_model(do_print=False)
                current_grad = imag_model.grad[ticker]
                if np.sign(np.diff(current_grad)[-1]) > 0 and break_values[0] is None:
                    break_values[0] = value
                elif np.sign(np.diff(current_grad)[-2]) > 0 and break_values[1] is None:
                    break_values[1] = value
                if all(break_values):
                    break_values_dict[ticker] = break_values
            
        self._print_issue('INFO', 'Break values: {}'.format(break_values_dict))

###############################################################################
#   INTERNAL FUNCTIONS
###############################################################################
    def _check_ticker_input(self, tickers):
        if isinstance(tickers, str):
            return [tickers]
        elif isinstance(tickers, list):
            return tickers
        else:
            raise TypeError('[ERROR]: Input of "tickers" must either be "str" or "list".')

    def _calc_ema(self, data, average_sample):
        '''
        Function to calculate the exponential moving average
        Inputs:
            - average_sample: number of days to average
        Outputs:
            - pandas.core.series.Series, i.e. ema prices.
        '''
        return data.ewm(span=average_sample, adjust=False).mean()

    def _init_model(self, periods=(12, 26, 9), *args, **kwargs):
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
        try:
            do_print = kwargs['do_print']
        except KeyError:
            do_print = True
        self._print_issue('INIT', 'Initialising model for tickers: {}'.format(self.tickers), \
                          do_print=do_print)
        macd = self._calc_ema(self.data, periods[0]) - \
               self._calc_ema(self.data, periods[1])
        signal_line = self._calc_ema(macd, periods[2])
        grad = np.gradient(macd - signal_line)
        local_min, local_max, grad_dict = {}, {}, {}
        if isinstance(grad, list):
            self._print_issue('WARNING', 'Ignoring second entry of gradient!', \
                              do_print=do_print)
            grad = grad[0].T
            for n in range(grad.shape[0]):
                local_min[self.tickers[n]] = argrelextrema(grad[n], np.less)
                local_max[self.tickers[n]] = argrelextrema(grad[n], np.greater)
        else:
            local_min[self.tickers[0]] = argrelextrema(grad, np.less)[0]
            local_max[self.tickers[0]] = argrelextrema(grad, np.greater)[0]
        #transforming grad as dict
        if len(grad.shape) == 1:
            grad_dict[self.tickers[0]] = grad
        else:
            for n, ticker in enumerate(self.tickers):
                grad_dict[ticker] = grad[n]

        self.local_min = local_min
        self.local_max = local_max
        self.grad = grad_dict
        self._print_issue('INIT', 'Successfully initialized model.', \
                          do_print=do_print)
        self._print_issue(None, '*' * 82, \
                          do_print=do_print)

    def _get_locs(self, ticker):
        buy_locs = self.local_min[ticker][0] + self.buy_delay
        sell_locs = self.local_max[ticker][0] + self.buy_delay
        try:
            if buy_locs[0] > sell_locs[0]:
                sell_locs = sell_locs[1:]
        except IndexError:
            self._print_issue('INFO', 'First sell position will not be displayed.')
        #check locs:
        if buy_locs.shape[0] > sell_locs.shape[0]:
            self._print_issue('INFO', 'Open position.')
        elif buy_locs.shape[0] < sell_locs.shape[0]:
            try:
                sell_locs[0] = buy_locs[0]
            except IndexError:
                self._print_issue('INFO', 'No buy locations occured.\
    Sell locations are set to buy locations.')
                sell_locs = buy_locs
        return buy_locs, sell_locs
###############################################################################
#   USEFUL FUNCTIONS
###############################################################################
    def _print_issue(self, key, issue, do_print=True):
        if do_print:
            if key is not None:
                print('[{}]: {}'.format(key, issue))
            else:
                print('{}'.format(issue))
