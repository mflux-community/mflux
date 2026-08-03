import functools
import logging
import threading

import mlx.core as mx

log = logging.getLogger(__name__)

_main_thread_id = threading.main_thread().ident


def with_gpu_stream(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if threading.current_thread().ident == _main_thread_id:
            return fn(*args, **kwargs)
        stream = mx.new_stream(mx.gpu)
        with mx.stream(stream):
            return fn(*args, **kwargs)

    return wrapper
