from tools import colors, load_csv, parse_csv
import sys as sys

def get():
    try:
        if len(sys.argv) != 2:
            raise AssertionError("no file name specified.")
    except AssertionError as error:
        print(colors.clr.fg.red, "Error:", error, colors.clr.reset)
        sys.exit(1)
    path = sys.argv[1]
    data = load_csv.load(path) # check file and get brut csv data
    csv_object = parse_csv.parse(data, path) # parse csv data and get a csv_object

    return csv_object

def get_no_parse():
    try:
        if len(sys.argv) != 2:
            raise AssertionError("no file name specified.")
    except AssertionError as error:
        print(colors.clr.fg.red, "Error:", error, colors.clr.reset)
        sys.exit(1)
    path = sys.argv[1]
    data = load_csv.load(path) # check file and get brut csv data

    return data