import time


def timestamp_to_human(timestamp):
    human_readable_time = time.strftime('%H:%M:%S%f', time.localtime(timestamp))
    return human_readable_time


def timed_func(func):
    """
    Decorator to time a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(f"Time elapsed for {func.__name__}: {end_time - start_time}s")

        return result

    return wrapper


def timed_hoc(func, *args, **kwargs):
    """
    Higher order function to time a function.
    :param func:
    :param args:
    :param kwargs:
    :return:
    """

    try:
        name = func.__name__  # Works for functions
    except AttributeError:
        try:
            name = func.module.__class__.__name__  # Works for nn.Module objects
        except AttributeError:
            name = 'model'  # Default name if above methods fail

    start_time = time.time()
    print(f"Start time for {name}: {timestamp_to_human(start_time)}")
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time elapsed for {name}: {end_time - start_time}s")

    return result
