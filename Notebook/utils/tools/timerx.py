import time

class Timer:
    '''
    代码的计时
    '''
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("%s took: %0.3f seconds" % (self.name, self.interval))
        return False

def timer(func):
    '''
    定义一个计时器，传入一个，并返回另一个附加了计时功能的方法
    函数修饰器
    '''
    # 定义一个内嵌的包装函数，给传入的函数加上计时功能的包装
    def wrapper(*args, **kw):
        start = time.time()
        print('Loading in memory ...')
        value = func(*args, **kw)
        end = time.time()
        print('used time: {0:g} s'.format(end - start))
        return value

    # 将包装后的函数返回
    return wrapper