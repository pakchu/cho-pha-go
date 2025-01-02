import datetime

# wrapper function to measure time
def timeit(method):
    def timed(*args, **kw):
        ts = datetime.datetime.now()
        result = method(*args, **kw)
        te = datetime.datetime.now()
        print(f'Method: {method.__name__}, From: {ts}, To: {te}, Time: {te - ts}')
        return result
    return timed