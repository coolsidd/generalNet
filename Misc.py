from functools import wraps
import logging
import time


def logged(orig_func):
    logging.basicConfig(filename=orig_func.__name__, level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info("{} With args {} and kwargs {}".format(
            orig_func.__name__, args, kwargs))
        return orig_func(*args, **kwargs)
    return wrapper


def timed(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print("{} ran in {}".format(orig_func.__name__, t2))
        return result
    return wrapper


def debug(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        print("{} is about to run with args {} and kwargs {}".format(
            orig_func.__name__, args, kwargs))
    return wrapper
