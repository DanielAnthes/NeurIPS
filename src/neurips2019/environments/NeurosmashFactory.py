from neurips2019.environments.EnvironmentFactory import EnvironmentFactory
from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment

class NeurosmashFactory(EnvironmentFactory):

    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1):
        super().__init__()
        self.ip = ip
        self.port = port
        self.size = size
        self.timescale = timescale

    def get_instance(self):
        with self.num_instances.get_lock():
            self.num_instances.value +=1
            # for now the first neurosmash instance gets port 13000, port numbers are incremented for following instances
            port = self.port + (self.num_instances.value - 1)
            env = NeurosmashEnvironment(timescale=self.timescale, size=self.size, port=port, ip=self.ip)
        return env
