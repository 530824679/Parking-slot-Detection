import re
import os
import time
import glob
import math
import logging
import random
import numpy as np
from pathlib import Path


# Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

# Search for file if not found
def check_file(file):
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file

# Return path to most recent 'last.ckpt' in /runs (i.e. to --resume from)
def get_latest_run(search_dir='.'):
    last_list = glob.glob(f'{search_dir}/**/last*.ckpt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''

# Increments a directory runs/exp1 --> runs/exp2_comment
def increment_dir(dir, comment=''):
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')

# Returns x evenly divisible by divisor
def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

# Verify img_size is a multiple of stride s
def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size