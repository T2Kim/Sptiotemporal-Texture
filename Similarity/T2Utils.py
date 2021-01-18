import time


def watchModule(timing_module, timing_all):
    def watch(func):
        def watch_f(*args, **kwargs):
            if timing_module or timing_all:
                start = time.time()
            result = func(*args, **kwargs)
            if timing_module or timing_all:
                end = time.time()
                print("%s %s %f" % (func.__name__, "elapse time:", end - start))
            return result
        return watch_f
    return watch