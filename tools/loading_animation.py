import itertools
import time
import sys
import threading


def loading():
    global loading_str
    global load

    for c in itertools.cycle(['|', '/', '-', '\\']):
        if load:
            break
        sys.stdout.write('\033[33m\r ' + loading_str + "  " + c + "  ")
        sys.stdout.flush()
        time.sleep(0.1)

def stop():
    global load
    global t_load

    load = True
    sys.stdout.write(chr(13))
    sys.stdout.write(' '*100)
    sys.stdout.write(chr(13))
    print('\033[33m\r ' + loading_str, '\033[0m\r')

def start(string: str):
    global load
    global loading_str
    global t_load

    loading_str = string
    load = False
    t_load = threading.Thread(target=loading)
    t_load.start()