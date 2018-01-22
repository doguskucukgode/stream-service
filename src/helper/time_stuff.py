import time
import signal

# Simple Python decorator to measure execution time of methods
def measure_time(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("Function '"+ f.__name__ +"' took: "+ str((end-start)*1000) +" ms")
        return result
    return wrapper

# A simple custom timeout exception implemented just for the 'add_timeout' decorator below
class CustomTimeoutException(Exception):
    pass

# A decorator that adds timeout logic to the given function
def add_timeout(timeout):
    def call_with_timeout(timeout, f, *args, **kwargs):
        """Call f with the given arguments, but if timeout seconds pass before f
        returns, raise CustomTimeoutException. The exception is raised async, so
        data structures being updated by f may be in an inconsistent state.
        """
        def handler(signum, frame):
            raise CustomTimeoutException("Timed out after {} seconds.".format(timeout))
        old = signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(timeout)
            try:
                return f(*args, **kwargs)
            finally:
                signal.alarm(0)
        finally:
            signal.signal(signal.SIGALRM, old)

    def the_decorator(f):
        def wrapper(*args, **kwargs):
            return call_with_timeout(timeout, f, *args, **kwargs)
        return wrapper
    return the_decorator
