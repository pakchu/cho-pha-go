import datetime
import functools
import operator
from collections import defaultdict

time_records = defaultdict(datetime.timedelta)

# wrapper function to measure time
def timeit(method):
    def timed(*args, **kw):
        ts = datetime.datetime.now()
        result = method(*args, **kw)
        te = datetime.datetime.now()
        print(f'Method: {method.__name__}, From: {ts}, To: {te}, Time: {te - ts}')
        return result
    return timed

def time_record(method):
    def timed(*args, **kw):
        ts = datetime.datetime.now()
        result = method(*args, **kw)
        te = datetime.datetime.now()
        time_records[method.__name__] += te - ts
        return result
    return timed

def product(iterable):
    "Like sum(), but with multiplication."
    return functools.reduce(operator.mul, iterable)