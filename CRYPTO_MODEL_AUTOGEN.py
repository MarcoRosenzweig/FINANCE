#!/usr/bin/env python
# coding: utf-8

# # FROM TERMINAL, RUN THIS LINE:
# jupyter nbconvert --execute --to html CRYPTO_MODEL_AUTOGEN.ipynb

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999')


# In[ ]:


from model import MODEL
import utils, plotting, fstats
import pandas as pd


# In[ ]:


#your intra-day time at which to evaluate the model.
day_hour = 18
#your tickers of interest
tickers = ['BTC-USD']


# In[ ]:


#do not edit below this cell!


# In[ ]:


#specify dates:
todays_date = pd.Timestamp.today()
start_date = todays_date - pd.Timedelta('200 days')
filter_date = start_date.floor(freq='D').replace(hour=day_hour)
#get data:
model = MODEL(tickers=tickers)
model.get_data(start=start_date, interval='60m')
#filter by datetime:
date_range = utils.create_date_range(start_date=filter_date)
model.apply_date_filter(date_range, force_apply=True)


# In[ ]:


model.eval_model()


# In[ ]:


plot_date = todays_date - pd.Timedelta('30 days')
plot_start = str(plot_date.date())


# In[ ]:


plotting.plot_model(model, tickers='BTC-USD', plot_from_date=plot_start)


# In[ ]:


imag_model = model.copy_model()
imag_model.append_timedelta(timedelta=1)
imag_model.comp_break_values(tickers='all', parallel_computing=True)
imag_model._init_model()


# In[ ]:


imag_model.show_possibilities(plot_from_date=plot_start, switch_axes=False)


# In[ ]:


fstats.calc_probs(model=imag_model, tickers='all', auto_update_tolerances=True)

