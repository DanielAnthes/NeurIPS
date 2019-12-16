from neurips2019.environments.EnvironmentFactory import EnvironmentFactory
from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment

class NeurosmashFactory(EnvironmentFactory):

    def get_instance(self, timescale=1, size=768, ip="127.0.0.1"):
        with self.num_instances.get_lock():
            self.num_instances.value +=1
            # for now the first neurosmash instance gets port 13000, port numbers are incremented for following instances
            port = 13000 + (self.num_instances - 1)
            env = NeurosmashEnvironment(timescale=timescale, size=size, port=port, ip=ip)
        return env
