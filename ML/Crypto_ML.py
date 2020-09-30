import sys, os, logging
from argparse import ArgumentParser
import time
from model import MODEL


parser = ArgumentParser()

parser.add_argument("-t", "--tickers", 
                    type=str,
                    dest="tickers", 
                    action="store",
                    nargs="+",
                    required=True,
                    help="Tickers to evaluate. Can be multiple.")

parser.add_argument("-es", "--evaluationStyle", 
                    type=int,
                    dest="evaluationStyle",
                    action="store",
                    nargs=1,
                    default="hourly",
                    choices=["hourly", "daily"],
                    help="Evaluate data hourly or daily.")

parser.add_argument("-sl", "--sequenceLength", 
                    type=int,
                    dest="sequenceLength",
                    action="store",
                    nargs=1,
                    default=14,
                    help="Sequence length of prior data.")

parser.add_argument("-uc", "--useColumns",
                    type=str,
                    dest="useColumns",
                    action="store",
                    default=["Close"],
                    choices=["Open", "High", "Low", "Close", "Adj Close", "Volume"],
                    help="Columns of data to use.")

def main(args):
    """Main function."""
    
    tickers = args["tickers"]
    
#===============================================================================
# CALLING MAIN:
#===============================================================================
if __name__ == "__main__":
    #parsing arguments from command line:
    args = parser.parse_args(sys.argv[1:])
    #setting up timer:
    START_TIME = time.time()
    START_MESSAGE = "=" * 78
    START_MESSAGE += "\nSTARTING ML ANALYSIS...\n"
    START_MESSAGE += "=" * 78
    print(START_MESSAGE)
    #executing main function:
    main(vars(args))
    #print elapsed time:
    ELAPSED_TIME = time.time() - START_TIME
    SUCCES_MESSAGE = "\n\n"
    SUCCES_MESSAGE += "=" * 78
    SUCCES_MESSAGE += "\nML ANALYSIS SUCCESSFULLY TERMINATED AFTER {:.2f}sec.".format(ELAPSED_TIME)
    print(SUCCES_MESSAGE)