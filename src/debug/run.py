from A3C import A3C
from torch.multiprocessing import Queue
from Logger import Logger
from threading import Thread

queue = Queue()
logger = Logger("logs/cp4", queue)
log_thread = Thread(target=logger.run, name="logger")
agent = A3C(queue)

try:
    log_thread.start()
    agent.train(6, 4000) # train with 6 processes
    print("FINISHED TRAINING")
    agent.evaluate(10)
    # stop logger
    queue.put(None)
    log_thread.join()

except KeyboardInterrupt as e:
    # if interrupt collect thread first
    queue.put(None)
    log_thread.join()
    raise e

