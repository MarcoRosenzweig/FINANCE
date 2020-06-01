#!/usr/bin/env python
# coding: utf-8

from model import MODEL
import utils, plotting, statistics
import pandas as pd


# In[ ]:


start_date = '2020-01-01' #your start date for the model.
day_hour = 18 #your intra-day time at which to evaluate the model.
tickers = 'BTC-USD'


# In[ ]:


model = MODEL(tickers=tickers)
model.get_data(start=start_date, interval='60m')
model.data.tail(2)


# In[ ]:


start_date_range = pd.Timestamp(2020, 1, 1, day_hour)
date_range = utils.create_date_range(start_date=start_date_range)
model.apply_date_filter(date_range, force_apply=True)
model.data.tail()


# In[ ]:


model.eval_model()


# In[ ]:


plotting.plot_model(model, tickers='BTC-USD', plot_from_date='2020-04-01')


# In[ ]:


imag_model = model.copy_model()
imag_model.append_timedelta(timedelta=1)
imag_model.comp_break_values(tickers='all', parallel_computing=True)
imag_model._init_model()


# In[ ]:


imag_model.show_possibilities(plot_from_date='2020-05-04', switch_axes=False)


# In[ ]:


statistics.calc_probs(model=imag_model, tickers='all', auto_update_tolerances=True)

