import sys as sys
from tools import colors, load_csv, parse_csv, get_csv_object
from math import sqrt, ceil



if __name__ == "__main__":

    csv_object = get_csv_object.get()

    print(csv_object["data_std"])    

    print("Done for test.")
    print("")
    exit(0)