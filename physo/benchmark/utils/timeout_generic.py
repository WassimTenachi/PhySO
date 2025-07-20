import multiprocessing as mp
import queue
import platform
import signal
from functools import wraps
from contextlib import contextmanager
from typing import Callable, Any
import time

class TimeoutError(Exception):
    pass


if platform.system() != "Windows":
    # Unix/macOS (signal-based) - Compatible with all Python versions
    @contextmanager
    def _timeout_unix(seconds: float):
        def handler(signum, frame):
            raise TimeoutError(f"Function timed out after {seconds}s")

        original_handler = signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(int(seconds))  # Using alarm() for broader compatibility
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)


    def _run_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
        with _timeout_unix(timeout):
            return func(*args, **kwargs)

else:
    # Windows (process-based)
    def _run_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
        result_queue = mp.Queue()
        error_queue = mp.Queue()

        def worker():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                error_queue.put(e)

        proc = mp.Process(target=worker, daemon=True)
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise TimeoutError(f"Function timed out after {timeout}s")

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            raise error_queue.get_nowait()


def timeout(timeout_sec: float) -> Callable:
    """Decorator to add timeout to any function.
    Example usage:
    @timeout(2)  # 2
    def slow_function():
        time.sleep(4)
        return 'hi'

    slow_function() # This will raise a TimeoutError after 2 seconds
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapped(*args, **kwargs) -> Any:
            return _run_with_timeout(func, timeout_sec, *args, **kwargs)

        return wrapped

    return decorator


