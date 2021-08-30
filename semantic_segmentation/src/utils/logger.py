import sys
import time
import paddle

levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
log_level = 2

def log(level=2, message=""):
    if paddle.distributed.ParallelEnv().local_rank == 0:
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if log_level >= level:
            print("{} [{}]\t{}".format(
                current_time, 
                levels[level], 
                message).encode("utf-8").decode("latin1"))
            sys.stdout.flush()

def debug(message=""):
    log(level=3, message=message)

def info(message=""):
    log(level=2, message=message)

def warning(message=""):
    log(level=1, message=message)

def error(message=""):
    log(level=0, message=message)
