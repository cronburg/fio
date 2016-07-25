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
    parser.add_argument('-bw', '--bandwidth', dest='bandwidth', action='store_true', default=True, help='input contains bandwidth log files (default).')
    parser.add_argument('-lat', '--latency', dest='latency', action='store_true', default=False, help='input contains latency log files (defaults to -bw).')
    parser.add_argument("FILE", help="collectl log output files to parse", nargs="+")
    args = parser.parse_args()
    return args

class Interval():
    def __init__(self, ctx, start, end, series):
        self.ctx = ctx
        self.start = start
        self.end = end
        self.series = series

    def __repr__(self):
        return "Interval(start=%d, end=%d, series=%s)" \
            % (self.start, self.end, repr(self.series))

    def get_samples(self):
        return [item for sublist in [ts.get_samples(self.start, self.end) for ts in self.series] for item in sublist]

    def get_min(self):
        return min([sample.value for sample in self.get_samples()])

    def get_max(self):
        return max([sample.value for sample in self.get_samples()])

    def get_wa(self, samples, weight):
        total = 0
        for sample in samples:
            total += sample.value * sample.get_weight(self.start, self.end)
        return total / weight

    def get_wa_list(self):
        samples_list = [ts.get_samples(self.start, self.end) for ts in self.series]
        return [self.get_wa(samples, 1) for samples in samples_list]

    def get_wa_sum(self):
        return sum(self.get_wa_list())

    def get_wa_avg(self):
        total_values  = 0.0
        total_weights = 0.0
        total_samples = 0
        for ts in self.series:
            samples = ts.get_samples(self.start, self.end)
            total_samples += len(samples)
            for sample in samples:
                weight = sample.get_weight(self.start, self.end)
                total_weights += weight
                total_values  += sample.value * weight
        return total_values / total_weights

    def get_wp(self, p):
        """ Weighted percentile computation for the requested percentile (parameter p).
            Using a "Weighted Nearest Rank" method. Formulas for Sn, pn, and v are as
            described here: https://en.wikipedia.org/wiki/Percentile#Weighted_percentile

            Formulas:
                Sn :: n-th partial sum of the sample weights
                pn :: weighted percent rank of the nth value in our samples
                v  :: linearly interpolated value of the P-th percentile
            
            Notes:
                - 0th   percentile is defined to be the smallest value
                - 100th percentile is defined to be the largest  value
        """
        
        # Nested closures are used to more closely match the mathematical definitions
        Sn = lambda ws: lambda n: sum(ws[:n+1])
        _pn = lambda ws: lambda n: 100 / Sn(ws)(len(ws) - 1) * (Sn(ws)(n) - ws[n] / 2.0)
        def v(vs,ws):
            pn = _pn(ws)
            def perc(P):
                for k,p in enumerate(map(pn, range(len(ws)))):
                    if p > P:
                        k = k - 1
                        if k == -1: return vs[0]
                        return vs[k] + (P - pn(k)) / (pn(k+1) - pn(k)) * (vs[k+1] - vs[k])
                return vs[-1]
            return perc

        samples = self.get_samples()
        samples.sort(key=lambda x: x.value)
        values  = map(lambda s: s.value, samples)
        weights = map(lambda s: s.get_weight(self.start, self.end), samples)
        return v(values, weights)(p)

    @staticmethod
    def get_ftime(series):
        ftime = 0
        for ts in series:
            if ftime == 0 or ts.last.end < ftime:
                ftime = ts.last.end
        return ftime

    @staticmethod
    def get_intervals(series, itime):
        intervals = []
        ftime = Interval.get_ftime(series)
        start = 0
        end = itime
        while (start < ftime):
            intervals.append(Interval(ctx, start, end, series))
            start += itime
            end += itime
        return intervals

class TimeSeries():

    USEC_TO_MSEC = 1000.0

    def __init__(self, ctx, fn):
        self.ctx = ctx
        self.last = None 
        self.samples = []
        self.read_data(fn)

    def __repr__(self):
        return "TimeSeries(last=%s, samples=%s)" \
            % (repr(self.last), repr(self.samples))

    def read_data(self, fn):
        f = open(fn, 'r')
        p_time = 0
        for line in f:
            (time, value, _, _) = map(int, line.rstrip('\r\n').rsplit(', '))
            if   self.ctx.latency:
                self.add_sample(time - value / TimeSeries.USEC_TO_MSEC, time, value)
            elif self.ctx.bandwidth:
                if p_time == time:
                    raise ValueError("Error: sample start time and end time are same (%d). " % (time,) \
                                   + "Did you mean to use option '-lat' for latency files?")
                self.add_sample(p_time, time, value)
            p_time = time
 
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
        if start == end:
            raise ValueError("Error: sample start time and end time are same (%d)" % (start,))
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
        return float(ebound-sbound) / (self.end - self.start)

    def __repr__(self):
        return "Sample(start=%d, end=%d, value=%d)" \
                % (self.start, self.end, self.value)

class Printer():
    def __init__(self, ctx, series):
        self.ctx = ctx
        self.series = series
        self.ffmt = "%0.3f"

    def format(self, data):
        if isinstance(data, float) or isinstance(data, int):
            data = data / self.ctx.divisor
            return self.ffmt % data
        return data

    def print_full(self):
        for i in Interval.get_intervals(self.series, ctx.interval):
            value = ', '.join(self.format(j) for j in i.get_wa_list())
            print "%s, %s" % (self.ffmt % i.end, value)

    def print_sums(self):
        for i in Interval.get_intervals(self.series, ctx.interval):
            print "%s, %s" % (self.ffmt % i.end, self.format(i.get_wa_sum()))


    def print_averages(self):
        for i in Interval.get_intervals(self.series, ctx.interval):
            print "%s, %s" % (self.ffmt % i.end, self.format(i.get_wa_avg()))

    def print_all_stats(self):
        print('end-time, samples, min, avg, median, 90%, 95%, 99%, max')
        for i in Interval.get_intervals(self.series, ctx.interval):
            print(', '.join([
                "%d" % i.end,
                "%d" % len(i.get_samples()),
                self.format(i.get_min()),
                self.format(i.get_wa_avg()),
                self.format(i.get_wp(50)),
                self.format(i.get_wp(90)),
                self.format(i.get_wp(95)),
                self.format(i.get_wp(99)),
                self.format(i.get_max())
            ]))

    def print_default(self):
        interval = Interval.get_intervals(self.series, Interval.get_ftime(series))[0]
        print self.format(interval.get_wa_sum())


if __name__ == '__main__':
    ctx = parse_args()
    series = []
    for fn in ctx.FILE:
        series.append(TimeSeries(ctx, fn))

    p = Printer(ctx, series)

    if ctx.sum:
        p.print_sums()
    elif ctx.average:
        p.print_averages()
    elif ctx.full:
        p.print_full()
    elif ctx.allstats:
        p.print_all_stats()
    else:
        p.print_default()

