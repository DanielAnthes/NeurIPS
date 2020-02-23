from A3C import A3C
from torch.multiprocessing import Queue
from Logger import Logger
from threading import Thread


def main():
    queue = Queue()
    logger = Logger("logs/cp_pxl_5", queue)
    log_thread = Thread(target=logger.run, name="logger")
    agent = A3C(queue)

    try:
        log_thread.start()
        agent.train(4, 5000) # train with 4 processes
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

if __name__ == "__main__":
    main() 


