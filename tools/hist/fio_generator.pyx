""" Cythonizing this gave an ~20% overall speedup. The call to min()
    was the main culprit, which includes the str.split() and int()
    (6.4 seconds out of a 35.5 second run).
"""
import sys
import re
err = sys.stderr.write
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
    
