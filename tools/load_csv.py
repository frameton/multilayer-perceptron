import pandas as pd
import os
from tools import colors


def load(path: str) -> pd.core.frame.DataFrame:
    """
    Read a csv file with pandas
    """
    try:
        if not os.path.exists(path):
            raise AssertionError("file not found.")
        if not path.lower().endswith('.csv'):
            raise AssertionError("only csv file supported.")
        data = pd.read_csv(path, header=0)
        return data
    except AssertionError as error:
        print(colors.clr.fg.red, "Error:", error, colors.clr.reset)
        exit(1)





# csv_fileh = open(path, 'rb')
        #try:
            #dialect = csv.Sniffer().sniff(csv_fileh.read(1024))
            # Perform various checks on the dialect (e.g., lineseparator,
            # delimiter) to make sure it's sane

            # Don't forget to reset the read position back to the start of
            # the file before reading any entries.
           #csv_fileh.seek(0)
        #except Exception:
            #print(colors.clr.fg.red, "Error: only csv file supported.", colors.clr.reset)
            #exit(1)