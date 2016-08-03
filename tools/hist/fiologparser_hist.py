#!/usr/bin/env python2.7
""" Converts file(s) given on command line, assumed to be fio histogram files,
    into their corresponding differenced files i.e. non-cumulative histograms
    because fio outputs cumulative histograms, but we want histograms corresponding
    to individual time intervals

    Example usage:
    
        $ fiologparser-hist.py *_clat_hist*
        end-time, samples, min, avg, median, 90%, 95%, 99%, max
        1000, 15, 192.000, 1678.107, 1788.859, 1856.076, 1880.040, 1899.208, 1888.000
        2000, 43, 152.000, 1642.368, 1714.099, 1816.659, 1845.552, 1888.131, 1888.000
        ...

    @author Karl Cronburg <karl.cronburg@gmail.com>
"""
import sys
import pandas
import numpy as np
from fiologparser_numpy import weights, columns, weighted_percentile \
                            , weighted_average, percs, fmt_float_list

# Default values - see beginning of main() for how we detect number columns in
# the input files:
__HIST_COLUMNS = 1216
__NON_HIST_COLUMNS = 3
__TOTAL_COLUMNS = __HIST_COLUMNS + __NON_HIST_COLUMNS
    
def sequential_diffs(head_row, times, rws, hists):
    """ TODO: Make this code less disgusting in terms of numpy appending """
    result = np.empty(shape=(0, __HIST_COLUMNS))
    result_times = np.empty(shape=(1, 0))
    for i in range(8):
        idx = np.where(rws == i)
        diff = np.diff(np.append(head_row[i], hists[idx], axis=0), axis=0).astype(int)
        result = np.append(diff, result, axis=0)
        result_times = np.append(times[idx], result_times)
    idx = np.argsort(result_times)
    return result[idx]
    #np.diff(np.append(head_row, hists, axis=0), axis=0).astype(int)

def read_chunk(head_row, rdr, sz):
    """ Read the next chunk of size sz from the given reader, computing the
        differences across neighboring histogram samples.
    """
    try:
        """ StopIteration occurs when the pandas reader is empty, and AttributeError
            occurs if rdr is None due to the file being empty. """
        new_arr = rdr.read().values
    except (StopIteration, AttributeError):
        return None    

    """ Extract array of just the times, and histograms matrix without times column.
        Then, take the sequential difference of each of the rows in the histogram
        matrix. This is necessary because fio outputs *cumulative* histograms as
        opposed to histograms with counts just for a particular interval. """
    times, rws, szs = new_arr[:,0], new_arr[:,1], new_arr[:,2]
    hists = new_arr[:,__NON_HIST_COLUMNS:]
    hists_diff   = sequential_diffs(head_row, times, rws, hists)
    times = times.reshape((len(times),1))
    arr = np.append(times, hists_diff, axis=1)

    """ hists[-1] will be the row we need to start our differencing with the
        next time we call read_chunk() on the same rdr """
    return arr, hists[-1]

def get_min(fps, arrs):
    """ Find the file with the current first row with the smallest start time """
    return min([fp for fp in fps if not arrs[fp] is None], key=lambda fp: arrs.get(fp)[0][0][0])

def histogram_generator(ctx, fps, sz):
    
    """ head_row for a particular file keeps track of the last (cumulative)
        histogram we read. """
    head_row  = np.zeros(shape=(1, __HIST_COLUMNS))
    head_rows = {fp: {i: head_row for i in range(8)} for fp in fps}

    # Create a chunked pandas reader for each of the files:
    rdrs = {}
    for fp in fps:
        try:
            rdrs[fp] = pandas.read_csv(fp, dtype=int, header=None, chunksize=sz)
        except ValueError as e:
            if e.message == 'No columns to parse from file':
                if not ctx.nowarn: sys.stderr.write("WARNING: Empty input file encountered.\n")
                rdrs[fp] = None
            else:
                raise(e)

    # Initial histograms (and corresponding
    arrs = {fp: read_chunk(head_rows[fp], rdr, sz) for fp,rdr in rdrs.items()}
    while True:

        try:
            """ ValueError occurs when nothing more to read """
            fp = get_min(fps, arrs)
        except ValueError:
            return
        arr, head_row = arrs[fp]
        yield np.insert(arr[0], 1, fps.index(fp))
        arrs[fp] = arr[1:], head_row
        head_rows[fp] = head_row

        if arrs[fp][0].shape[0] == 0:
            arrs[fp] = read_chunk(head_rows[fp], rdrs[fp], sz)

def plat_idx_to_val(idx, edge=0.5, FIO_IO_U_PLAT_BITS=6, FIO_IO_U_PLAT_VAL=64):
    """ Taken from fio's stat.c for calculating the latency value of a bin
        from that bin's index. """

    # MSB <= (FIO_IO_U_PLAT_BITS-1), cannot be rounded off. Use
    # all bits of the sample as index
    if (idx < (FIO_IO_U_PLAT_VAL << 1)):
        return idx 

    # Find the group and compute the minimum value of that group
    error_bits = (idx >> FIO_IO_U_PLAT_BITS) - 1 
    base = 1 << (error_bits + FIO_IO_U_PLAT_BITS)

    # Find its bucket number of the group
    k = idx % FIO_IO_U_PLAT_VAL

    # Return the mean (if edge=0.5) of the range of the bucket
    return base + ((k + edge) * (1 << error_bits))
    
def plat_idx_to_val_coarse(idx, coarseness, edge=0.5):
    # Multiply the index by the power of 2 coarseness to get the bin
    # bin index with a max of 1536 bins (FIO_IO_U_PLAT_GROUP_NR = 24 in stat.h)
    stride = 1 << coarseness
    idx = idx * stride
    lower = plat_idx_to_val(idx, edge=edge)
    upper = plat_idx_to_val(idx + stride, edge=edge)
    return lower + (upper - lower) * edge

def print_all_stats(ctx, end, ss_cnt, mn, vs, ws, mx):
    ps = weighted_percentile(percs, vs, ws)

    avg = weighted_average(vs, ws)
    values = [mn, avg] + list(ps) + [mx]
    row = [end, ss_cnt] + map(lambda x: float(x) / ctx.divisor, values)
    fmt = "%d, %d, " + fmt_float_list(ctx, 7)
    print (fmt % tuple(row))
        
def update_extreme(val, fncn, new_val):
    """ Calculate min / max in the presence of None values """
    if val is None: return new_val
    else: return fncn(val, new_val)

# See beginning of main() for how bin_vals are computed
bin_vals = []
lower_bin_vals = [] # lower edge of each bin
upper_bin_vals = [] # upper edge of each bin 

def process_interval(ctx, samples, iStart, iEnd):
    """ Construct the weighted histogram for the given interval by scanning
        through all the histograms and figuring out which of their bins have
        samples with latencies which overlap with the given interval.
    """
    
    times, files, hists = samples[:,0], samples[:,1], samples[:,2:]
    iHist = np.zeros(__HIST_COLUMNS)
    ss_cnt = 0 # number of samples affecting this interval
    mn_bin_val, mx_bin_val = None, None

    for end_time,file,hist in zip(times,files,hists):
            
        # Only look at bins of the current histogram sample which
        # started before the end of the current time interval [start,end]
        start_times = (end_time - 0.5 * ctx.interval) - bin_vals / 1000.0
        idx = np.where(start_times < iEnd)
        s_ts, l_bvs, u_bvs, hs = start_times[idx], lower_bin_vals[idx], upper_bin_vals[idx], hist[idx]

        # Increment current interval histogram by weighted values of future histogram:
        ws = hs * weights(s_ts, end_time, iStart, iEnd)
        iHist[idx] += ws
    
        # Update total number of samples affecting current interval histogram:
        ss_cnt += np.sum(hs)

        # Update min and max bin values seen if necessary:
        idx = np.where(hs != 0)[0]
        if idx.size > 0:
            mn_bin_val = update_extreme(mn_bin_val, min, l_bvs[idx][0])
            mx_bin_val = update_extreme(mx_bin_val, max, u_bvs[idx][-1])

    if ss_cnt > 0: print_all_stats(ctx, iEnd, ss_cnt, mn_bin_val, bin_vals, iHist, mx_bin_val)

def main(ctx):

    # Automatically detect how many columns are in the input files,
    # calculate the corresponding 'coarseness' parameter used to generate
    # those files, and calculate the appropriate bin latency values:
    with open(ctx.FILE[0], 'r') as fp:
        global bin_vals,lower_bin_vals,upper_bin_vals,__HIST_COLUMNS,__TOTAL_COLUMNS
        fp.readline()
        __TOTAL_COLUMNS = len(fp.readline().split(', '))
        __HIST_COLUMNS = __TOTAL_COLUMNS - __NON_HIST_COLUMNS
        # TODO: FIO_IO_U_PLAT_GROUP_NR should be used to determine max_cols
        if __HIST_COLUMNS % 1216 == 0: max_cols = 1216.0
        else: max_cols = 1536.0
        coarseness = int(np.log2(max_cols / __HIST_COLUMNS))
        bin_vals = np.array(map(lambda x: plat_idx_to_val_coarse(x, coarseness), np.arange(__HIST_COLUMNS)), dtype=float)
        lower_bin_vals = np.array(map(lambda x: plat_idx_to_val_coarse(x, coarseness, 0.0), np.arange(__HIST_COLUMNS)), dtype=float)
        upper_bin_vals = np.array(map(lambda x: plat_idx_to_val_coarse(x, coarseness, 1.0), np.arange(__HIST_COLUMNS)), dtype=float)

    fps = [open(f, 'r') for f in ctx.FILE]
    gen = histogram_generator(ctx, fps, ctx.buff_size)

    print(', '.join(columns))

    try:
        start, end = 0, ctx.interval
        arr = np.empty(shape=(0,__TOTAL_COLUMNS - 1))
        more_data = True
        while more_data or len(arr) > 0:
            
            # Read up to 5 minutes of data from end of current interval.
            while len(arr) == 0 or arr[-1][0] < ctx.max_latency * 1000 + end:
                try:
                    new_arr = next(gen)
                except StopIteration:
                    more_data = False
                    break
                arr = np.append(arr, new_arr.reshape((1,__TOTAL_COLUMNS - 1)), axis=0)
            arr = arr.astype(int)
            
            if arr.size > 0:
                process_interval(ctx, arr, start, end)
                
                # Update arr to throw away samples we no longer need - samples which
                # end before the start of the next interval, i.e. the end of the
                # current interval:
                idx = np.where(arr[:,0] > end)
                arr = arr[idx]
            
            start += ctx.interval
            end = start + ctx.interval
    finally:
        map(lambda f: f.close(), fps)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    arg = p.add_argument
    arg("FILE", help='space separated list of latency log filenames', nargs='+')
    arg( '--buff_size'
       , default=10000
       , type=int
       , help='number of samples to buffer into numpy at a time'
       )
    arg( '--max_latency'
       , default=20
       , type=float
       , help='number of seconds of data to process at a time'
       )
    arg( '-i', '--interval'
       , default=1000
       , type=int
       , help='interval width (ms)'
       )
    arg( '-d', '--divisor'
       , required=False
       , type=int
       , default=1
       , help='divide the results by this value.'
       )
    arg( '--decimals'
       , default=3
       , type=int
       , help='number of decimal places to print floats to'
       )
    arg( '--nowarn'
       , dest='nowarn'
       , action='store_false'
       , default=True
       , help='do not print any warning messages to stderr'
       )
       
    main(p.parse_args())

