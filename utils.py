import yfinance as yf
import pandas as pd

def download_data(value='Close', *args, **kwargs):
    '''
    Function to get historical asset data of yahoo finance.
    Inputs:
        - *args, **kwargs: see yf.download docstring!
    Outputs:
        - pandas DataFrame of data.
    '''
    if value is None:
        return yf.download(*args, **kwargs)
    return yf.download(*args, **kwargs)[value]

def create_date_range(start_date, end_date=None, \
                      freq='d', tz='Europe/London', name='Datetime', \
                      *args, **kwargs):
    '''
    Function to generate a pandas date range.
    Inputs:
        - start_date: start date of date range. Must be compatible with
        pandas date_range function. Error is raised if not.
        - end_date: end date of date range. Must be compatible with
        pandas date_range function. Error is raised if not.
            default: None -> today.
        - args, kwargs: additional arguments, which must be compatible with
        pandas date_range function. Error is raised if not.
    Returns:
        - pandas date range
    '''
    #freq='d', tz='Europe/London', name='Datetime'
    if end_date is None:
        end_date = pd.Timestamp.today()
    return pd.date_range(start=start_date, end=end_date, \
                         freq=freq, tz=tz, name=name, \
                         *args, **kwargs)

def print_opening(do_print=True, *args, **kwargs):
    '''
    Function to print opening of algorithm.
    Inputs:
        - args and kwargs: both will be printed, but "_" in args will be replaced by " ".
    '''
    n_break_chars = 82
    lines = []
    lines.append('=' * n_break_chars)
    header = ['PRICE MODEL']
    version = ['Version 0.3']
    copyright = ['Authors: Marco Rosenzweig & Patrick Lorenz']
    for text in [header, version, copyright]:
        lines.append(next(map(_format_string, text)))
    lines.append('-' * n_break_chars)
    for arg, kwarg in kwargs.items():
        arg = arg.replace('_', ' ')
        if not isinstance(kwarg, str):
            kwarg = str(kwarg)
        text = [(' = ').join([arg, kwarg])]
        lines.append(next(map(_format_string, text)))
    lines.append('=' * n_break_chars)
    if do_print:
        for line in lines:
            print(line)

def print_closing(do_print=True, *args, **kwargs):
    '''
    Function to print closing of algorithm.
    Inputs:
        - args and kwargs: both will be printed, but "_" in args will be replaced by " ".
    '''
    n_break_chars = 82
    if do_print:
        print('-' * n_break_chars)

def _format_string(string, center_alligned=True):
    '''
    Internal function used for print_opening and print_closing
    '''
    special_char = '|'
    if center_alligned:
        string = string.center(80)
    return '{}{}{}'.format(special_char, string, special_char)

def check_ticker_input(tickers_input, tickers_avail, do_print=True):
    if isinstance(tickers_input, str):
        tickers_input = [tickers_input]
    elif isinstance(tickers_input, list):
        tickers_input = tickers_input
    else:
        raise TypeError('[ERROR]: Input of "tickers_input" must either be \
"str" or "list".')
    if tickers_avail is None:
        return tickers_input
    valid_tickers = []
    for ticker in tickers_input:
        if ticker not in tickers_avail:
            _print_issue('WARNING', 'Ticker "{}" not in available \
tickers'.format(ticker), do_print=do_print)
        else:
            valid_tickers.append(ticker)
    if len(valid_tickers) == 0:
        raise OSError('[DEPP]: No input ticker in self.tickers.')
    return valid_tickers

def _print_issue(key, issue, do_print=True):
    if do_print:
        if key is not None:
            print('[{}]: {}'.format(key, issue))
        else:
            print('{}'.format(issue))
