import yfinance as yf
import pandas as pd

def download_data(value=None, *args, **kwargs):
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
