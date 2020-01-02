from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import numpy as np
from enum import Enum


class LogType(Enum):
    DISK=0
    SCALAR=1
    HISTOGRAM=2
    IMAGE=3

LogEntry = namedtuple('LogEntry', ['type', 'key', 'value', 'idx', 'kwargs'])

class Logger:
    '''
    Class that manages logging statistics and makes them accessible to Tensorboard,
    utility for weight logging, saving agents
    '''
    def __init__(self, directory, queue):
        path = os.path.abspath(directory) # in case working directory changes elsewhere keep logging location
        if not os.path.isdir(path):
            os.makedirs(path)
        
        self.disk_log = dict()
        self.tb_log = dict()
        self.tb_writer = SummaryWriter(path)
        self.directory = path
        self.q = queue

    def run(self):
        type_switch = {
            LogType.DISK : lambda a,b,c,d: None,
            LogType.SCALAR : self.tb_writer.add_scalar,
            LogType.HISTOGRAM : self.tb_writer.add_histogram,
            LogType.IMAGE : self.tb_writer.add_image
        }

        while True:
            entry = self.q.get()
            if entry is None:
                self.tb_writer.close()
                break
            func = type_switch[entry.type]
            func(entry.key, entry.value, global_step=entry.idx, **entry.kwargs)

    def log_tb(self, key, value, idx=None, walltime=None):
        # log to dict to be written to tensorboard
        if not key in self.tb_log.keys():
            self.tb_log[key] = list()
        self.tb_log[key].append((value, idx, walltime))

    def log_disk(self, key, value, idx=None):
        # log to dict that will be written to disk
        if not key in self.disk_log.keys():
            self.disk_log[key] = list()
        self.disk_log[key].append((value,idx))

    def write(self):
        # write logs to tensorboard
        # with SummaryWriter(log_dir=self.directory) as writer:
        #     for key, hist in self.tb_log.items():
        #         for (value, idx, walltime) in hist:
        #             writer.add_scalar(tag=key, scalar_value=value, global_step=idx, walltime=walltime)
        # write logs to disk
        path = os.path.join(self.directory, f"logs_{time.time()}")
        with open(path, 'w') as file:
            file.write(json.dumps(self.disk_log))


