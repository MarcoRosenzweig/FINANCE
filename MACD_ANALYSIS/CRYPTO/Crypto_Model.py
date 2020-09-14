import sys, os, logging
from argparse import ArgumentParser
import time
from model import MODEL
import utils, plotting, fstats_pct_chg
import pandas as pd

#===============================================================================
# PARSING
#===============================================================================
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

parser.add_argument("-of", "--outputFolder",
                    type=str,
                    dest="outputFolder",
                    action="store",
                    nargs=1,
                    default=None,
                    help="Output folder to store outputs.")

parser.add_argument("-sf", "--saveFigures",
                    dest="saveFigures",
                    action="store_true",
                    help="Save figures into outputFolder.")

parser.add_argument("-wlf", "--writeLogFile",
                    dest="writeLogFile",
                    action="store_true",
                    help="Write log file of messages.")

parser.add_argument("-wme", "--writeModelEvaluation",
                    dest="writeModelEvaluation",
                    action="store_true",
                    help="Write model evaluation as csv.")

#===============================================================================
# UTILS:
#===============================================================================

def _get_as_str(inst, get=str, return_index=0):
    if not isinstance(inst, get):
        if isinstance(inst, list):
            return inst[return_index]
        return get(inst)
    return inst

def create_folder(folder_name):
    os.makedirs(name=folder_name, exist_ok=True)
    
#===============================================================================
# MAIN:
#===============================================================================

def main(args):
    """Main function."""
    
    tickers = args["tickers"]
    start_date_delta = args["startDateDelta"]
    plot_date_delta = args["plotDateDelta"]
    day_hour = args["dayHour"]
    entry_money = _get_as_str(args["entryMoney"], str, 0)
    #set plotting:
    save_figures = args["saveFigures"]
    if save_figures:
        return_plot = False
    else:
        return_plot = True
    #prepare output_folder:
    todays_date = pd.Timestamp.today()
    output_folder = _get_as_str(args["outputFolder"], str)
    if output_folder is not None:
        if output_folder == "auto":
            output_folder = "OUTPUT/{}".format(todays_date.strftime("%Y_%m_%d"))
        create_folder(output_folder)
    else:
        return_plot = True
        
    #set log file:
    log_file = args["writeLogFile"]
    if log_file:
        if output_folder is not None:
            log_file = "{}/log.txt".format(output_folder)
        else:
            log_file = "log.txt"
        logging.basicConfig(filename=log_file, level=logging.INFO)
        logger = logging.getLogger()
        sys.stderr.write = logger.error
        sys.stdout.write = logger.info
    else:
        #TODO: decide what happens here:
        pass
    do_print = True    
    #main function starts here:
    #specify dates:
    start_date = todays_date - pd.Timedelta("{} days".format(start_date_delta))
    filter_date = start_date.floor(freq="D").replace(hour=day_hour)
    #get data:
    model = MODEL(tickers=tickers)
    model.get_data(start=start_date, interval="60m")
    #filter by datetime:
    date_range = utils.create_date_range(start_date=filter_date)
    model.apply_date_filter(date_range, force_apply=True, do_print=do_print)
    #eval model:
    model.eval_model(tickers=tickers, entry_money=entry_money, do_print=do_print)
    #plotting:
    plot_date = todays_date - pd.Timedelta("{} days".format(plot_date_delta))
    plot_start = str(plot_date.date())
    plotting.plot_model(model, tickers=tickers,
                        plot_from_date=plot_start,
                        return_plot=return_plot,
                        save_figures=save_figures,
                        output_folder=output_folder,
                        do_print=do_print)
    #imag_model:
    imag_model = model.copy_model()
    imag_model.append_timedelta(timedelta=1, do_print=do_print)
    #compute break values:
    imag_model.comp_break_values(tickers="all", 
                                 parallel_computing=True, 
                                 do_print=do_print)
    #init model:
    imag_model._init_model(do_print=do_print)
    #show possibilities:
    imag_model.show_possibilities(plot_from_date=plot_start, 
                                  switch_axes=False, 
                                  return_plot=return_plot,
                                  save_figures=save_figures,
                                  output_folder=output_folder,
                                  do_print=do_print)
    #calulate statistics:
    fstats_pct_chg.calc_probs(model=imag_model, 
                              tickers="all", 
                              auto_update_tolerances=True,
                              return_plot=return_plot,
                              save_figures=save_figures,
                              output_folder=output_folder)
    #disable logger here:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    

#===============================================================================
# CALLING MAIN:
#===============================================================================
if __name__ == "__main__":
    #parsing arguments from command line:
    args = parser.parse_args(sys.argv[1:])
    #setting up timer:
    START_TIME = time.time()
    START_MESSAGE = "=" * 78
    START_MESSAGE += "\nSTARTING MACD ANALYSIS...\n"
    START_MESSAGE += "=" * 78
    print(START_MESSAGE)
    #executing main function:
    main(vars(args))
    #print elapsed time:
    ELAPSED_TIME = time.time() - START_TIME
    SUCCES_MESSAGE = "\n\n"
    SUCCES_MESSAGE += "=" * 78
    SUCCES_MESSAGE += "\nMACD ANALYSIS SUCCESSFULLY TERMINATED AFTER {:.2f}sec.".format(ELAPSED_TIME)
    print(SUCCES_MESSAGE)
    
    