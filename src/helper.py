import time

# Simple Python decorator to measure execution time of methods
def measure_time(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("Function '"+ f.__name__ +"' took: "+ str((end-start)*1000) +" ms")
        return result
    return wrapper
