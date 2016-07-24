#!/usr/bin/env python2.7
""" Converts file(s) given on command line, assumed to be fio histogram files,
    into their corresponding differenced files i.e. non-cumulative histograms
    because fio outputs cumulative histograms, but we want histograms corresponding
    to individual time intervals

    Example usage:
    
        $ fiologparser-hist.py *_clat_hist*

    @author Karl Cronburg <karl.cronburg@gmail.com>
"""
import pandas
import numpy as np
from fiologparser_numpy import weights, columns, weighted_percentile, percs, fmt_float_list

def read_chunk(head_row, rdr, sz):
    """ Read the next chunk of size sz from the given reader, computing the
        differences across neighboring histogram samples.
    """
    try:
        new_arr = rdr.read().values
    except StopIteration:
        return None    
    
    times, hists = new_arr[:,0], new_arr[:,1:]
    hists_diff   = np.diff(np.append(head_row, hists, axis=0), axis=0).astype(int)
    times = times.reshape((len(times),1))
    arr = np.append(times, hists_diff, axis=1)
    return arr, hists[-1]

def get_min(fps, arrs):
    """ Find the file with the current first row with the smallest start time """
    return min([fp for fp in fps if not arrs[fp] is None], key=lambda fp: arrs.get(fp)[0][0][0])

def histogram_generator(fps, sz):
    #head_row = np.fromstring(fp.readline(), sep=',')
    head_row  = np.zeros(shape=(1,1216))
    head_rows = {fp: head_row for fp in fps}
    rdrs = {fp: pandas.read_csv(fp, dtype=int, header=None, chunksize=sz) for fp in fps}
    arrs = {fp: read_chunk(head_row, rdr, sz) for fp,rdr in rdrs.items()}
    while True:

        try:
            fp = get_min(fps, arrs)
        except ValueError:
            return
        arr, head_row = arrs[fp]
        yield np.insert(arr[0], 1, fps.index(fp))
        arrs[fp] = arr[1:], head_row
        
        if arrs[fp][0].shape[0] == 0:
            #import pdb; pdb.set_trace()
            arrs[fp] = read_chunk(head_rows[fp], rdrs[fp], sz)

def plat_idx_to_val(idx, FIO_IO_U_PLAT_BITS=6, FIO_IO_U_PLAT_VAL=64):
    # MSB <= (FIO_IO_U_PLAT_BITS-1), cannot be rounded off. Use
    # all bits of the sample as index
    if (idx < (FIO_IO_U_PLAT_VAL << 1)):
        return idx 

    # Find the group and compute the minimum value of that group
    error_bits = (idx >> FIO_IO_U_PLAT_BITS) - 1 
    base = 1 << (error_bits + FIO_IO_U_PLAT_BITS)

    # Find its bucket number of the group
    k = idx % FIO_IO_U_PLAT_VAL

    # Return the mean of the range of the bucket
    return base + ((k + 0.5) * (1 << error_bits))
    
def print_all_stats(ctx, end, ss_cnt, mn, vs, ws, mx):
    ps = weighted_percentile(percs, vs, ws)

    values = [mn, np.sum(vs * ws) / ss_cnt] + list(ps) + [mx]
    row = [end, ss_cnt] + map(lambda x: float(x) / ctx.divisor, values)
    fmt = "%d, %d, " + fmt_float_list(ctx, 7)
    print (fmt % tuple(row))

    #import pylab as p; p.hist(iHist, np.log2(bin_vals)); p.show()
        
def update_extreme(val, fncn, new_val):
    if val is None: return new_val
    else: return fncn(val, new_val)

bin_vals = np.array(map(plat_idx_to_val, np.arange(1216)), dtype=float)
def process_interval(ctx, samples, iStart, iEnd):
    
    times, files, hists = samples[:,0], samples[:,1], samples[:,2:]
    iHist = np.zeros(1216)
    ss_cnt = 0 # number of samples affecting this interval
    mn_bin_val, mx_bin_val = None, None

    for end_time,file,hist in zip(times,files,hists):
            
        # Only look at bins of the current histogram sample which
        # started before the end of the current time interval [start,end]
        start_times = (end_time - 0.5 * ctx.interval) - bin_vals / 1000.0
        idx = np.where(start_times < iEnd)
        s_ts, bvs, hs = start_times[idx], bin_vals[idx], hist[idx]

        # Increment current interval histogram by weighted values of future histogram:
        ws = hs * weights(s_ts, end_time, iStart, iEnd)
        iHist[idx] += ws
    
        # Update total number of samples affecting current interval histogram:
        ss_cnt += np.sum(hs)

        # Update min and max bin values seen if necessary:
        #import pdb; pdb.set_trace()
        idx = np.where(hs != 0)[0]
        if idx.size > 0:
            mn_bin_val = update_extreme(mn_bin_val, min, bvs[idx][0])
            mx_bin_val = update_extreme(mx_bin_val, max, bvs[idx][-1])

    print_all_stats(ctx, iEnd, ss_cnt, mn_bin_val, bin_vals, iHist, mx_bin_val)

def main(ctx):
    fps = [open(f, 'r') for f in ctx.FILE]
    gen = histogram_generator(fps, ctx.buff_size)

    print(', '.join(columns))

    try:
        start, end = 0, ctx.interval
        arr = np.empty(shape=(0,1218))
        more_data = True
        while more_data or len(arr) > 0:
            
            # Read up to 5 minutes of data from end of current interval.
            while len(arr) == 0 or arr[-1][0] < ctx.max_latency * 1000 + end:
                try:
                    new_arr = next(gen)
                except StopIteration:
                    more_data = False
                    break
                arr = np.append(arr, new_arr.reshape((1,1218)), axis=0)
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
       , default=300
       , type=float
       , help='number of seconds of data to process at a time'
       )
    arg( '-i', '--interval'
       , default=100
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
    main(p.parse_args())

