import sys
from argparse import ArgumentParser
import time
from model import MODEL
import utils, plotting, fstats_pct_chg
import pandas as pd

parser = ArgumentParser()

parser.add_argument("-t", "--tickers", 
                    type=str,
                    dest="tickers", 
                    action="store",
                    nargs="+",
                    required=True,
                    help="Tickers to evaluate. Can be multiple.")

parser.add_argument("-sdd", "--startDateDelta", 
                    type=int,
                    dest="startDateDelta",
                    action="store",
                    nargs=1,
                    default=200,
                    help="Time delta to start the evaluation. Default: 200 days prior to today.")

parser.add_argument("-pdd", "--plotDateDelta", 
                    type=int,
                    dest="plotDateDelta",
                    action="store",
                    nargs=1,
                    default=31,
                    help="Time delta to plot the evaluation. Default: 30 days prior to today.")

parser.add_argument("-dh", "--dayHour",
                    type=int,
                    dest="dayHour",
                    action="store",
                    nargs=1,
                    default=18,
                    help="Hour of day to evaluate model. This can be dangerous for assets in different time zones.")

parser.add_argument("-em", "--entryMoney",
                    type=float,
                    dest="entryMoney",
                    action="store",
                    nargs=1,
                    default=200.0,
                    help="Entry money to grow.")

#parser.add_argument()

def main(args):
    #setting parsed arguments:
    tickers = args["tickers"]
    start_date_delta = args["startDateDelta"]
    plot_date_delta = args["plotDateDelta"]
    day_hour = args["dayHour"]
    entry_money = args["entryMoney"]
    #main function starts here:
    #specify dates:
    todays_date = pd.Timestamp.today()
    start_date = todays_date - pd.Timedelta("{} days".format(start_date_delta))
    filter_date = start_date.floor(freq="D").replace(hour=day_hour)
    #get data:
    model = MODEL(tickers=tickers)
    model.get_data(start=start_date, interval="60m")
    #filter by datetime:
    date_range = utils.create_date_range(start_date=filter_date)
    model.apply_date_filter(date_range, force_apply=True)
    #eval model:
    model.eval_model(tickers=tickers, entry_money=entry_money)
    #plotting:
    plot_date = todays_date - pd.Timedelta("{} days".format(plot_date_delta))
    plot_start = str(plot_date.date())
    plotting.plot_model(model, tickers=tickers, plot_from_date=plot_start, return_plot=True)
    imag_model = model.copy_model()
    imag_model.append_timedelta(timedelta=1)
    imag_model.comp_break_values(tickers="all", parallel_computing=True)
    imag_model._init_model()
    imag_model.show_possibilities(plot_from_date=plot_start, switch_axes=False, return_plot=True)
    fstats_pct_chg.calc_probs(model=imag_model, tickers="all", auto_update_tolerances=True)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    START_TIME = time.time()
    START_MESSAGE = "=" * 78
    START_MESSAGE += "\nSTARTING MACD_ANALYSIS...\n"
    START_MESSAGE += "=" * 78
    print(START_MESSAGE)
    #executing main function:
    main(vars(args))
    ELLAPSED_TIME = time.time() - START_TIME
    SUCCES_MESSAGE = "=" * 78
    SUCCES_MESSAGE += "\n\nMACD_ANALYSIS SUCCESSFULLY TERMINATED AFTER {:.2f}sec.".format(ELLAPSED_TIME)
    print(SUCCES_MESSAGE)
    
    