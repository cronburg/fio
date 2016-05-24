#!/usr/bin/python
#
# fiologparser.py
#
# This tool lets you parse multiple fio log files and look at interaval
# statistics even when samples are non-uniform.  For instance:
#
# fiologparser.py -s *bw*
#
# to see per-interval sums for all bandwidth logs or:
#
# fiologparser.py -a *clat*
#
# to see per-interval average completion latency.

import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interval', required=False, type=int, default=1000, help='interval of time in seconds.')
    parser.add_argument('-d', '--divisor', required=False, type=int, default=1, help='divide the results by this value.')
    parser.add_argument('-f', '--full', dest='full', action='store_true', default=False, help='print full output.')
    parser.add_argument('-A', '--all', dest='allstats', action='store_true', default=False, 
                        help='print all stats for each interval.')
    parser.add_argument('-a', '--average', dest='average', action='store_true', default=False, help='print the average for each interval.')
    parser.add_argument('-s', '--sum', dest='sum', action='store_true', default=False, help='print the sum for each interval.')
    parser.add_argument("FILE", help="collectl log output files to parse", nargs="+")
    args = parser.parse_args()

    return args

def get_ftime(series):
    ftime = 0
    for ts in series:
        if ftime == 0 or ts.last.end < ftime:
            ftime = ts.last.end
    return ftime

def print_full(ctx, series):
    ftime = get_ftime(series)
    start = 0 
    end = ctx.interval

    while (start < ftime):
        end = ftime if ftime < end else end
        samples_array = [ts.get_samples(start, end) for ts in series]
        wa = [weighted_average(samples, start, end, 1) for samples in samples_array]
        print "%s, %s" % (end, ', '.join(["%0.3f" % i for i in wa]))
        start += ctx.interval
        end += ctx.interval

def print_sums(ctx, series):
    ftime = get_ftime(series)
    start = 0
    end = ctx.interval

    while (start < ftime):
        end = ftime if ftime < end else end
        samples_array = [ts.get_samples(start, end) for ts in series]
        total = sum([weighted_average(samples, start, end, 1) for samples in samples_array])
        print "%s, %0.3f" % (end, total)
        start += ctx.interval
        end += ctx.interval

def print_averages(ctx, series):
    ftime = get_ftime(series)
    start = 0
    end = ctx.interval

    while (start < ftime):
        end = ftime if ftime < end else end
        samples_array = [ts.get_samples(start, end) for ts in series]
        samples = [item for sublist in samples_array for item in sublist]
        print "%s, %0.3f" % (end, weighted_average(samples, start, end, len(series)))
        start += ctx.interval
        end += ctx.interval

def print_all_stats(ctx, series):
    ftime = get_ftime(series)
    start = 0 
    end = ctx.interval
    print('start-time, samples, min, avg, median, 90%, 95%, 99%, max')
    while (start < ftime):  # for each time interval
        end = ftime if ftime < end else end
        # compute all stats and print them
        samples_array = [ts.get_samples(start, end) for ts in series]
        samples = [item for sublist in samples_array for item in sublist]
        mymin = min([sample.value for sample in samples])
        myavg = weighted_average(samples, start, end, len(series)) 
        mymedian = weighted_percentile(samples, start, end, len(series), 0.5)
        my90th = weighted_percentile(samples, start, end, len(series), 0.90) 
        my95th = weighted_percentile(samples, start, end, len(series), 0.95)
        my99th = weighted_percentile(samples, start, end, len(series), 0.99)
        mymax = max([sample.value for sample in samples])
        print( '%f, %d, %f, %f, %f, %f, %f, %f, %f' % (
            end, len(samples), 
            mymin, myavg, mymedian, my90th, my95th, my99th, mymax))

        # advance to next interval
        start += ctx.interval
        end += ctx.interval

def weighted_average(samples, start, end, total_weight):
    total = 0
    for sample in samples:
        total += sample.value * sample.get_weight(start, end)
    return total / total_weight

def weighted_percentile(samples, start, end, total_weight, p):
    s=sorted(samples, key=lambda x: x.value)
    weight = 0
    last = None
    cur = None

    # first find the two samples that straddle the percentile based on weight
    for sample in s:
        if weight > total_weight * p:
           break
        weight += sample.get_weight(start, end) 
        last = cur
        cur = sample

    # next find the weighted average of those two samples
    lw = last.get_weight(start, end)
    cw = cur.get_weight(start, end)
    tw = lw + cw 
    return (last.value * lw + cur.value * cw) / tw 

def print_default(ctx, series):
    start = 0
    end = get_ftime(series) 

    samples_array = [ts.get_samples(start, end) for ts in series]
    total = sum([weighted_average(samples, start, end, 1) for samples in samples_array])
    print "%0.3f" % (total)

class TimeSeries():
    def __init__(self, ctx, fn):
        self.ctx = ctx
        self.last = None 
        self.samples = []
        self.read_data(fn)

    def read_data(self, fn):
        f = open(fn, 'r')
        p_time = 0
        for line in f:
            (time, value, foo, bar) = line.rstrip('\r\n').rsplit(', ')
            self.add_sample(p_time, int(time), int(value))
            p_time = int(time)
 
    def add_sample(self, start, end, value):
        sample = Sample(ctx, start, end, value)
        if not self.last or self.last.end < end:
            self.last = sample
        self.samples.append(sample)

    def get_samples(self, start, end):
        sample_list = []
        for s in self.samples:
            if s.get_weight(start, end) > 0:
                sample_list.append(s)
        return sample_list

class Sample():
    def __init__(self, ctx, start, end, value):
        self.ctx = ctx
        self.start = start
        self.end = end
        self.value = value

    def get_weight(self, start, end):
        # short circuit if not within the bound
        if (end < self.start or start > self.end):
            return 0
        sbound = self.start if start < self.start else start
        ebound = self.end if end > self.end else end
        return float(ebound-sbound) / (end-start)

if __name__ == '__main__':
    ctx = parse_args()
    series = []
    for fn in ctx.FILE:
       series.append(TimeSeries(ctx, fn)) 
    if ctx.sum:
        print_sums(ctx, series)
    elif ctx.average:
        print_averages(ctx, series)
    elif ctx.full:
        print_full(ctx, series)
    elif ctx.allstats:
        print_all_stats(ctx, series)
    else:
        print_default(ctx, series)

