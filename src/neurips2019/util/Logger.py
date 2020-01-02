from torch.utils.tensorboard import SummaryWriter
import os
import json
import time

class Logger:
    '''
    Class that manages logging statistics and makes them accessible to Tensorboard,
    utility for weight logging, saving agents
    '''
    def __init__(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.disk_log = dict()
        self.tb_log = dict()
        self.directory = directory

    def log_tb(self, key, value, idx=None, walltime=None):
        # log to dict to be written to tensorboard
        if not key in self.tb_log.keys():
            self.tb_log[key] = list()
        self.tb_log[key] = (value, idx, walltime)

    def log_disk(self, key, value, idx=None):
        # log to dict that will be written to disk
        if not key in self.disk_log.keys():
            self.weight_logger[key] = list()
        self.disk_log[key].append((value,idx))

    def write(self):
        # write logs to tensorboard
        writer = SummaryWriter(log_dir=self.directory)
        for key in self.tb_log.keys():
            for value, idx, time in self.tb_log.values():
                writer.add_scalar(tag=key, scalar_value=value, global_step=idx, walltime=time)
        # write logs to disk
        path = os.path.join(self.directory, f"logs_{time.time()}")
        with open(path, 'w') as file:
            file.write(json.dumps(self.disk_log))
