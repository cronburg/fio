#!/usr/bin/env python2.7
import sys

# Cut the number bins in half in fio histogram output

with open(sys.argv[1], 'r') as fp:
    for line in fp.readlines():
        vals = line.split(', ')
        sys.stdout.write("%s, %s, %s, " % tuple(vals[:3]))

        hist = list(map(int, vals[3:]))
        for i in range(0, len(hist) - 2, 2):
            sys.stdout.write("%d, " % (hist[i] + hist[i+1],))
        sys.stdout.write("%d\n" % (hist[len(hist) - 2] + hist[len(hist) - 1]))
        
