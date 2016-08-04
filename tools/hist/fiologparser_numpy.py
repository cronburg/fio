#!/usr/bin/env python2.7
""" NumPy + Pandas + Cython optimized streaming version of fiologparser.
    
    This tool parses multiple fio log files, computing interval statistics
    using weighted percentiles to account for samples spanning multiple
    intervals.

    Example usages:

        # Compute per-interval sums across all bandwidth files:
        $ fiologparser.py -s -bw /path/to/fio/logs/*_bw*

        # Compute all statistics for completion latency files:
        $ fiologparser.py -A -lat /path/to/fio/logs/*_clat*
    
    @author Karl Cronburg <karl@cronburg.com>
"""
import sys
import numpy as np
from numpy import genfromtxt, lexsort, around, where
land, lor, lnot = np.logical_and, np.logical_or, np.logical_not
from itertools import islice
import argparse
import os
from fiologparser_hist import weights, columns, weighted_percentile \
    , err, weighted_average, percs, fmt_float_list

try:
    import pandas
except ImportError:
    pass

try:
    import pyximport; pyximport.install()
    from fio_generator import fio_generator
except ImportError:
    """ User does not have Cython installed, or is otherwise missing the
        fio_generator.pyx file. Skip cythonizing by putting code directly here too: """

    import re
    row_re = re.compile("^\d+,\s\d+,\s\d+,\s\d+\s+$")

    def next_no_stop_iter(fp):
        try:
            while True:
                line = fp.next()
                if row_re.match(line):
                    return line
                err("WARNING: Ignoring line '%s' - does not match expected format.\n" % line.rstrip())
        except StopIteration:
            return None

    def get_time(line):
        try:
            return int(line.split(',')[0])
        except ValueError:
            return None

    def get_min(lines, fps):
        mn_fp, mn_time = None, None
        for fp in fps:
            line = lines.get(fp)
            if not line is None:
                time = get_time(line)
                if (not time is None) and (mn_time is None or time < mn_time):
                    mn_time, mn_fp = time, fp
        return mn_fp

    def fio_generator(fps):
        """ Create a generator for reading multiple fio files in end-time order """
        lines = {fp: next_no_stop_iter(fp) for fp in fps}

        while True:
            # Get fp with minimum value in the first column (fio log end-time value)
            fp = get_min(lines, fps)
            if fp is None: break
            yield (lines[fp].rstrip() + ", " + str(fps.index(fp)) + '\n')
            lines[fp] = next_no_stop_iter(fp) # read a new line into our dictionary

def print_sums(ctx, vs, ws, ss, end, divisor=1.0):
    fmt = "%d, " + fmt_float_list(ctx, 1)
    print (fmt % (end, weighted_average(vs, ws) / divisor / ctx.divisor))

def print_averages(ctx, vs, ws, ss, end):
    print_sums(ctx, vs, ws, ss, end, divisor=float(len(vs)))

def print_full(ctx, vs, ws, ss, end):
    fmt = "%d, " + fmt_float_list(ctx, len(ctx.FILE))

    # List of lists of where the last column in the samples is all the same
    # (corresponding to which file the input came from)
    idxs = [where(ss[:,-1] == i) for i in range(len(ctx.FILE))]
    
    # Each column in this row corresponds to the weighted sum of samples
    # falling in the current interval in a particular file:
    row = [np.sum(vs[idxs[i]] * ws[idxs[i]]) for i in range(len(ctx.FILE))]
    
    print (fmt % tuple([end] + row))

def print_all_stats(ctx, vs, ws, ss_cnt, end):
    ps = weighted_percentile(percs, vs, ws)

    values = [np.min(vs), weighted_average(vs, ws)] + list(ps) + [np.max(vs)]
    row = [end, ss_cnt] + map(lambda x: float(x) / ctx.divisor, values)
    fmt = "%d, %d, " + fmt_float_list(ctx, 7)
    print (fmt % tuple(row))

# TODO: not sure what the default is doing yet.
def print_default(ctx, vs, ws, ss, end):
    pass
    #print (fmt_float_list(ctx, 1) % (np.sum(vs * ws) / ctx.divisor,))

def process_interval(ctx, samples, start, end):
    """ Determine which of the given samples occur during the given interval,
        then compute and print the desired statistics - min, avg, percentiles, max.

        samples :: Array / matrix of samples (as seen in fio log files)
        start   :: int
        end     :: int
    """
    
    times,clats = samples[:,0], samples[:,1]
   
    if ctx.latency:
      start_times = times - (clats / 1000.0) # convert end time array to start times
    elif ctx.bandwidth:
      start_times = np.delete(np.insert(times, 0, 0), times.size)
    else:
      raise Exception("Please specify either --bandwidth or --latency.")

    # Sort by start time:
    idx = lexsort((start_times,))
    samples = samples[idx]
    times,clats = samples[:,0], samples[:,1]
    start_times = start_times[idx]

    # Determine which samples occured during the current interval [start,end]:
    idx = where(lnot(lor(start_times >= end, times <= start)))
    ss = samples[idx]
    start_ts = start_times[idx]
    end_ts, vs = ss[:,0], ss[:,1]
    
    if len(ss) > 0:
        ws = weights(start_ts, end_ts, start, end)
        
        if ctx.sum:         print_sums(ctx, vs, ws, ss, end)
        elif ctx.average:   print_averages(ctx, vs, ws, ss, end)
        elif ctx.full:      print_full(ctx, vs, ws, ss, end)
        elif ctx.allstats:  print_all_stats(ctx, vs, ws, len(ss), end)
        else:               print_default(ctx, vs, ws, ss, end)

def read_csv(ctx, fp, sz):
    try:
        gen = islice(fp, sz)
        try:
            data = pandas.read_csv(Reader(gen), dtype=int, header=None).values
        except NameError:
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                data = np.genfromtxt(gen, delimiter=',')
            if data.shape == (0,): raise ValueError()
    except ValueError:
        data = np.empty((0,5))
    return data

def read_next(ctx, fp, sz):
    data = read_csv(ctx, fp, sz)
    if len(data.shape) == 1:
        return np.array([data]) # Single-line file.
    return data

class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try:
            return next(self.g)
        except StopIteration:
            return ''

#iter_csv = pandas.read_csv('file.csv', iterator=True, chunksize=1000)
#df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])

def main(ctx):
    fps = [open(f, 'r') for f in ctx.FILE]
    fp = fio_generator(fps)
   
    if ctx.allstats:
        print(', '.join(columns))

    try:
        start = 0
        end = ctx.interval
        arr = read_next(ctx, fp, ctx.buff_size)
        more_data = True
        while more_data or len(arr) > 0:

            # Read up to 5 minutes of data from end of current interval.
            while len(arr) == 0 or arr[-1][0] < ctx.max_latency * 1000 + end:
                new_arr = read_next(ctx, fp, ctx.buff_size)
                if new_arr.shape[0] < ctx.buff_size:
                    more_data = False
                    arr = np.append(arr, new_arr, axis=0)
                    break
                arr = np.append(arr, new_arr, axis=0)

            if arr.size > 0:
                process_interval(ctx, arr, start, end)
        
                # Update arr to throw away samples we no longer need - samples which
                # end before the start of the next interval, i.e. the end of the
                # current interval:
                idx = where(arr[:,0] > end)
                arr = arr[idx]

            start += ctx.interval
            end = start + ctx.interval
        
    finally:
        map(lambda f: f.close(), fps)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    arg = p.add_argument
    arg( '--max_latency'
       , default=300
       , type=float
       , help='number of seconds of data to process at a time'
       )
    arg( '-i', '--interval'
       , default=10000
       , type=int
       , help='interval width (ms)'
       )
    arg( '-d', '--divisor'
       , required=False
       , type=int
       , default=1
       , help='divide the results by this value.'
       )
    arg( '-f', '--full'
       , dest='full'
       , action='store_true'
       , default=False
       , help='print full output.'
       )
    arg( '-A', '--all'
       , dest='allstats'
       , action='store_true'
       , default=False
       , help='print all stats for each interval.'
       )
    arg( '-a', '--average'
       , dest='average'
       , action='store_true'
       , default=False
       , help='print the average for each interval.'
       )
    arg( '-s', '--sum'
       , dest='sum'
       , action='store_true'
       , default=False
       , help='print the sum for each interval.'
       )
    arg( '--buff_size'
       , default=10000
       , type=int
       , help='number of samples to buffer into numpy at a time'
       )
    arg( '--decimals'
       , default=3
       , type=int
       , help='number of decimal places to print floats to'
       )
    arg( '-bw', '--bandwidth'
       , dest='bandwidth'
       , action='store_true'
       , default=False
       , help='input contains bandwidth log files.'
       )
    arg( '-lat', '--latency'
       , dest='latency'
       , action='store_true'
       , default=False
       , help='input contains latency log files.'
       )
    arg( "FILE"
       , help='space separated list of latency log filenames'
       , nargs='+'
       )
    main(p.parse_args())

