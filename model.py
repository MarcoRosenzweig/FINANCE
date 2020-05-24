import utils
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import copy
import plotting

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
    def __init__(self, tickers, data=None, buy_delay=1, periods=(12, 26, 9)):
        self.tickers = utils.check_ticker_input(tickers_input=tickers, \
                                                tickers_avail=None, \
                                                do_print=True)
        self.data = data
        self.local_min, self.local_max, self.grad = None, None, None
        self.buy_delay = buy_delay
        self.ticker_df = dict.fromkeys(self.tickers)
        self.break_values = None
        self.tolerances = None
        self.z_values = dict.fromkeys(self.tickers)
        self.periods = periods

    def get_data(self, value='Close', filter_date_range=None, *args, **kwargs):
        '''
        Function to get asset historical prices.
        '''
        self.data = utils.download_data(tickers=self.tickers, \
                                        value=value, \
                                        *args, **kwargs)
        #if only one ticker in self.tickers: download_data returns series!
        if not isinstance(self.data, pd.core.frame.DataFrame):
            self.data = self.data.to_frame(name=self.tickers[0])

        if filter_date_range is not None:
            self.apply_date_filter(filter_date_range=filter_date_range)

    def check_for_nan_values(self, tickers='all', exclude_last_value=True, \
                             *args, **kwargs):
        #TODO: CODE THIS FUNCTION !
        do_print = self._parse_kwargs('do_print', kwargs, error_arg=True)
        if tickers == 'all':
            tickers = self.tickers
        else:
            tickers = utils.check_ticker_input(tickers_input=tickers, \
                                               tickers_avail=self.tickers, \
                                               do_print=True)
        for ticker in tickers:
            if exclude_last_value:
                nan_indices = np.where(np.isnan(self.data[ticker][:-1]))[0]
                valid_indices = np.where(np.isfinite(self.data[ticker][:-1]))[0]
                valid_indices = np.hstack((valid_indices, self.data[ticker].shape[0] - 1))
                filtered_data = self.data[ticker][valid_indices]
                self.data[ticker] = filtered_data

            else:
                utils._print_issue('INFO', 'Last value is considered to be removed.', \
                                    do_print=do_print)
                nan_indices = np.where(np.isnan(self.data[ticker]))[0]
            if nan_indices.size > 0:

                #print(self.data[ticker].dropna())#[~np.isnan(self.data[ticker][:last_value_index])])
                return
                input_message = 'Remove {} NaN values? '.format(nan_indices.size)
                if self._get_answer(input_message=input_message):
                    self.data[ticker] = self.data[ticker][~nan_indices]
            else:
                utils._print_issue('INFO', 'No NaN values detected.', \
                                    do_print=do_print)

    def apply_date_filter(self, filter_date_range):
        try:
            filtered_data = self.data.reindex(filter_date_range)
        except KeyError:
            utils._print_issue('WARNING', 'filter not in data.')
            return
        else:
            #1 index is ticker, 0 index is data
            nan_values = np.where(np.isnan(filtered_data))[0]
            n_nan_values = nan_values.size
            if  n_nan_values > 0:
                utils._print_issue('WARNING', 'Filter would result in {} NaN values.'\
.format(n_nan_values))
                input_message = 'Remove NaN values?: '
                force_filter = self._get_answer(input_message=input_message)
                if force_filter:
                    filtered_data = filtered_data.dropna()
            self.data = filtered_data
            utils._print_issue('INFO', 'filter applied.')

    def eval_model(self, tickers='all', entry_money=200, fees=(1.0029, .9954), tax=.25, visualize=False, *args, **kwargs):
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
        do_print = self._parse_kwargs('do_print', kwargs, error_arg=True)
        if tickers == 'all':
            valid_tickers = self.tickers
        else:
            valid_tickers = utils.check_ticker_input(tickers_input=tickers, \
                                                     tickers_avail=self.tickers, \
                                                     do_print=True)
        utils.print_opening(ticker=valid_tickers, \
                            start_date=self.data.index[0].strftime('%D'), \
                            end_date=self.data.index[-1].strftime('%D'), \
                            initial_investment_per_ticker=entry_money, \
                            do_print=do_print)

        if any([self.local_min is None, self.local_max is None, self.grad is None]):
            self._init_model(do_print=do_print)

        for ticker in valid_tickers:
            utils._print_issue('TICKER', ticker, do_print=do_print)
            buy_locs, sell_locs = self._get_locs(ticker=ticker, \
                                                 do_print=do_print)
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
                utils._print_issue('INFO', '{:.2f} tax was paid.'.format(tax_pays), \
                                   do_print=do_print)
                net_income = (trade_rewards[-1] - entry_money) * (1 - tax)
            else:
                utils._print_issue('INFO', 'No tax paid.', \
                                   do_print=do_print)
                net_income = np.sum(trade_wins)
            #create final DataFrame
            sell_grad = self.grad[ticker][self.local_max[ticker]]
            buy_grad = self.grad[ticker][self.local_min[ticker]]
            #be aware that buy_dates can be 1 entry longer then sell dates!
            if buy_dates.shape[0] > sell_dates.shape[0]:
                if sell_dates.shape[0] > 0:
                    utils._print_issue('INFO', 'Last entry of "Sell Dates" will \
be assigned equally as the penultimate one.', do_print=do_print)
                    sell_dates = np.append(sell_dates, sell_dates[-1])
                else:
                    utils._print_issue('INFO', 'First entry of "Sell Dates" \
will be first entry of "Buy Dates".', do_print=do_print)
                    sell_dates = buy_dates[0]
                try:
                    sell_prices.loc[pd.Timestamp.max] = np.nan
                except: #OverflowError: --> NOT WORKING?
                    sell_prices.loc[buy_prices.index[-1]] = np.nan
                trade_rewards = np.append(trade_rewards, np.nan)
                trade_wins = np.append(trade_wins, np.nan)
                win_loss = np.append(win_loss, np.nan)
                sell_grad = np.append(sell_grad, np.nan)
            final_df = pd.DataFrame(data = {'Buy Dates': buy_dates, \
                                            'Sell Dates': sell_dates, \
                                            'Buy Prices': buy_prices.to_numpy(), \
                                            'Sell Prices': sell_prices.to_numpy(), \
                                            'Trade Reward': trade_rewards, \
                                            'Trade Win': trade_wins, \
                                            'Trade Efficiency': win_loss, \
                                            'Grad at Buy': buy_grad, \
                                            'Grad at Sell': sell_grad})
            self.ticker_df[ticker] = final_df
            utils._print_issue(None, '-' * 82, do_print=do_print)
            utils._print_issue('SUMMARY', \
                               'Average trade win: {:.10%}'.format(average_win), \
                               do_print=do_print)
            utils._print_issue('SUMMARY', \
                               'Average trade loss: {:.10%}'.format(average_loss), \
                               do_print=do_print)
            utils._print_issue('SUMMARY', \
                               'Efficiency: {:.2%}'.format(efficiency), \
                               do_print=do_print)
            utils._print_issue('SUMMARY', \
                               'NET WIN: {:.2f}'.format(net_income), \
                               do_print=do_print)
            utils._print_issue(None, '=' * 82, do_print=do_print)

    def copy_model(self):
        return copy.deepcopy(self)

    def append_timedelta(self, timedelta=1, overwrite_data=True, *args, **kwargs):
        do_print = self._parse_kwargs('do_print', kwargs, error_arg=True)
        new_entry = self.data.index[-1] + pd.Timedelta(days=timedelta)
        final_entries = list(self.data.index)
        final_entries.append(new_entry)
        idx = pd.DatetimeIndex(final_entries)
        new_data = self.data.reindex(idx)
        if overwrite_data:
            utils._print_issue('INFO', 'New data was appended.', \
                               do_print=do_print)
            self.data = new_data
        else:
            return new_data

    def comp_break_values(self, tickers='all', refactor_step_size=1, \
                          append_break_values=False, parallel_computing=True,\
                          *args, **kwargs):
        do_print = self._parse_kwargs('do_print', kwargs, error_arg=True)
        if tickers == 'all':
            tickers = self.tickers
        else:
            tickers = utils.check_ticker_input(tickers_input=tickers, \
                                               tickers_avail=self.tickers, \
                                               do_print=True)
        imag_model = self.copy_model()
        break_values_dict = dict.fromkeys(tickers)
        current_values = dict.fromkeys(tickers, None)
        tolerances = dict.fromkeys(tickers)
        deviation = .3
        utils._print_issue('INFO', 'Compute break values with {:.2%} deviation'.format(deviation), \
                           do_print=do_print)

        for ticker in tickers:
            utils._print_issue('INFO', 'Current ticker: {}'.format(ticker), \
                               do_print=do_print)
            break_values = [None, None]
            if np.isnan(self.data[ticker].values[-1]):
                value_index = -2
            else:
                value_index = -1
            current_values[ticker] = self.data[ticker].values[value_index]
            #create range:
            start_value = current_values[ticker] * (1 - deviation)
            end_value = current_values[ticker] * (1 + deviation)
            step_size = (current_values[ticker] / 5000) * refactor_step_size
            rng = np.arange(start_value, end_value, step_size)
            try:
                import multiprocessing as mp
            except ModuleNotFoundError:
                utils._print_issue('ERROR', 'Multiprocessing module not available.')
                parallel_computing = False
            if not parallel_computing:
                break_values_dict[ticker] = np.sort(self._comp_bvs(model=imag_model, \
                                                                   rng=rng, \
                                                                   ticker=ticker))
            else:
                n_procs = 10
                utils._print_issue('INFO', 'Using {} processes.'.format(n_procs))
                rng_list = self._do_array_split(rng, n_procs)
                from functools import partial
                inputs_partial = partial(self._comp_bvs, imag_model, ticker)
                with mp.Pool(processes=n_procs) as pool:
                    bvs = pool.map(inputs_partial, rng_list)
                bv_final = [None, None]
                for bv_list in bvs:
                    for n, bv in enumerate(bv_list):
                        if bv is not None and bv_final[n] is None:
                            bv_final[n] = bv
                        if all(bv_final):
                            break
                break_values_dict[ticker] = np.sort(bv_final)
            #make sure to already have sort break_values_dict!
            tolerances[ticker] = break_values_dict[ticker] - current_values[ticker]

        self.tolerances = tolerances
        self.break_values = break_values_dict
        if append_break_values:
            utils._print_issue('INFO', 'Appending break values to model data', \
                               do_print=do_print)
            for ticker in valid_tickers:
                smal_tol = np.argsort(tolerances[ticker])[0]
                self.data[ticker][-1] = break_values_dict[ticker][smal_tol]
                self._init_model(do_print=False)
        else:
            utils._print_issue('INFO', 'Current values: {}'.format(current_values), \
                              do_print=do_print)
            utils._print_issue('INFO', 'Break values: {}'.format(break_values_dict), \
                              do_print=do_print)
            utils._print_issue('INFO', 'Tolerances: {}'.format(tolerances), \
                              do_print=do_print)

    def show_possibilities(self, tickers='all', *args, **kwargs):
        if tickers == 'all':
            tickers = self.tickers
        else:
            tickers = utils.check_ticker_input(tickers_input=tickers, \
                                               tickers_avail=self.tickers, \
                                               do_print=True)
        for ticker in tickers:
            utils._print_issue(None, '=' * 82)
            utils._print_issue('INFO', 'Current ticker: {}'.format(ticker))
            #check if last value is nan:
            last_value_index = -1
            if not np.isnan(self.data[ticker][last_value_index]):
                utils._print_issue('WARNING', 'Last value of data set is not NaN!')
                input_message = 'Proceed anyways? '
                if not self._get_answer(input_message=input_message):
                    continue
            else:
                last_value_index = -2
            if self.break_values[ticker] is None and generic_value is None:
                utils._print_issue('ERROR', 'No break values computed for this ticker!')
                continue
            deviation = self._parse_kwargs('deviation', kwargs, error_arg=.0125)
            bottom_value, top_value = self.break_values[ticker]
            middle_value = (top_value - bottom_value)*.5 + bottom_value
            bottom_value *= (1 - deviation)
            top_value *= (1 + deviation)
            test_values = [bottom_value, middle_value, top_value]
            for value in test_values:
                utils._print_issue(None, '-' * 82)
                utils._print_issue('INFO', 'Result for value: {}'.format(value))
                #create an imag_model:
                test_model = self.copy_model()
                #assign the value to the last entry:
                test_model.data[ticker][-1] = value
                #init model
                test_model._init_model(do_print=False)
                test_model.eval_model(do_print=False)
                p_range = self._parse_kwargs('plot_range', kwargs, None)
                p_index = self._parse_kwargs('plot_from_index', kwargs, None)
                p_date = self._parse_kwargs('plot_from_date', kwargs, None)
                switch_axes = self._parse_kwargs('switch_axes', kwargs, False)
                plotting.plot_model(model=test_model, \
                                    tickers=ticker, \
                                    plot_range=p_range, \
                                    plot_from_index=p_index, \
                                    plot_from_date=p_date, \
                                    plot_break_values=True, \
                                    switch_axes=switch_axes)
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

    def _init_model(self, *args, **kwargs):
        '''
        Function to set up the price model. The idea is to locate the inflection
        points of the difference of "moving average converging diverging (macd)"
        and "Signal Line (signal_line)". These indicate local up and down trends.
        The actual buy and sell prices are therefore the next day, i.e. buy_delay.
        Inputs:
            - periods: days to calculate the macd (first two values)
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
        do_print = self._parse_kwargs('do_print', kwargs, error_arg=True)
        utils._print_issue('INIT', 'Initialising model for tickers: {}'.format(self.tickers), \
                          do_print=do_print)
        macd = self._calc_ema(self.data, self.periods[0]) - \
               self._calc_ema(self.data, self.periods[1])
        signal_line = self._calc_ema(macd, self.periods[2])
        if len(self.tickers) == 1:
            grad = np.gradient(macd[self.tickers[0]] - \
                               signal_line[self.tickers[0]])
        else:
            grad = np.gradient(macd - signal_line)
        local_min, local_max, grad_dict = {}, {}, {}
        if isinstance(grad, list):
            utils._print_issue('WARNING', 'Ignoring second entry of gradient!', \
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
        utils._print_issue('INIT', 'Successfully initialized model.', \
                          do_print=do_print)
        utils._print_issue(None, '*' * 82, \
                          do_print=do_print)

    def _get_locs(self, ticker, *args, **kwargs):
        do_print = self._parse_kwargs('do_print', kwargs, error_arg=True)
        if len(self.local_min) > 1:
            buy_locs = self.local_min[ticker][0] + self.buy_delay
            sell_locs = self.local_max[ticker][0] + self.buy_delay
        else:
            buy_locs = self.local_min[ticker] + self.buy_delay
            sell_locs = self.local_max[ticker] + self.buy_delay
        try:
            if buy_locs[0] > sell_locs[0]:
                sell_locs = sell_locs[1:]
        except IndexError:
            utils._print_issue('INFO', 'First sell position will not be displayed.', \
                               do_print=do_print)
        #check locs:
        if buy_locs.shape[0] > sell_locs.shape[0]:
            utils._print_issue('INFO', 'Open position.', do_print=do_print)
        elif buy_locs.shape[0] < sell_locs.shape[0]:
            try:
                sell_locs[0] = buy_locs[0]
            except IndexError:
                utils._print_issue('INFO', 'No buy locations occured.\
    Sell locations are set to buy locations.', do_print=do_print)
                sell_locs = buy_locs
        return buy_locs, sell_locs

    def _comp_bvs(self, model, ticker, rng):
        #make sure to be in consistent order of inputs w.r.t partial/pool.map function
        break_values = [None, None]
        for value in rng:
            model.data[ticker].values[-1] = value
            model._init_model(do_print=False)
            current_grad = model.grad[ticker]
            if np.sign(np.diff(current_grad)[-1]) > 0 and break_values[0] is None:
                break_values[0] = value
            elif np.sign(np.diff(current_grad)[-2]) > 0 and break_values[1] is None:
                break_values[1] = value
            if all(break_values):
                break
        return break_values

    def _do_array_split(self, rng, n_procs, safety_value_range=5):
        rngs = np.array_split(rng, n_procs)
        for n, rng in enumerate(rngs):
            if n != len(rngs) - 1:
                rngs[n] = np.append(rng, rngs[n + 1][:safety_value_range])
        return rngs
###############################################################################
#   USEFUL FUNCTIONS
###############################################################################
    def _check_ticker_input(self, tickers, do_print=True):
        if isinstance(tickers, str):
            tickers = [tickers]
        elif isinstance(tickers, list):
            tickers = tickers
        else:
            raise TypeError('[ERROR]: Input of "tickers" must either be "str" or "list".')

        if not hasattr(self, 'tickers'):
            return tickers

        valid_tickers = []
        for ticker in tickers:
            if ticker not in self.tickers:
                utils._print_issue('WARNING', 'Ticker "{}" not in self.tickers'.format(ticker), \
                                  do_print=do_print)
            else:
                valid_tickers.append(ticker)
        if len(valid_tickers) == 0:
            raise OSError('[DEPP]: No input ticker in self.tickers.')
        return valid_tickers

    def _get_answer(self, input_message, possibilities=['y', 'n']):
        answer = ''
        while answer not in possibilities:
            answer = input('[USER-INPUT]: {}'.format(input_message))
            if answer == 'y':
                return True
            elif answer == 'n':
                return False
            else:
                utils._print_issue('ERROR', 'Possible answers: {}.'.format(possibilities))

    def _parse_kwargs(self, key, kwargs, error_arg=False):
        try:
            return kwargs[key]
        except KeyError:
            return error_arg

class STATISTICAL_MODEL(MODEL):
    """docstring for STATISTICAL_MODEL."""

    def __init__(self, model):
        pass
