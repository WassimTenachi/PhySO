import errno
import os
import signal
import functools
import time

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """
    # Works on UNIX only
    # https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    Demo:
    @timeout(20)
    def myfunc(n):
        time.sleep(n)
        return True
    myfunc(n>20) will be killed
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

