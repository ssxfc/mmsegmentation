import os

from functools import wraps

from functools import wraps


class logit(object):
    """
    第一种定义方式（带有参数）：class
    """
    def __init__(self, logfile='out.log'):
        self.logfile = logfile

    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            print("记录日志信息.")
            self.notify()
            return func(*args, **kwargs)

        return wrapped_function

    def notify(self):
        pass


def assert_exists(fp):
    """
    第二种定义方式（带有参数）：general function
    Args:
        fp:

    Returns:

    """
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            assert os.path.exists(fp)
            return func(*args, **kwargs)
        return wrapped_function
    return logging_decorator


@assert_exists(fp=r"D:\py\engineering\mmsegmentation\demo\batch_mod_pixel.pyx")
def myfunc2():
    print("处理文件")


# if __name__ == "__main__":
#     myfunc2()