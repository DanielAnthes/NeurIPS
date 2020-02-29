import os

from A3C import A3C
from torch.multiprocessing import Queue
from Logger import Logger
from threading import Thread


def main():
    queue = Queue()
    logger = Logger("logs/cp_stack_long3", queue)
    log_thread = Thread(target=logger.run, name="logger")
    agent = A3C(queue)
    # agent.load(sorted(os.listdir("model"), key=lambda x: int(x), reverse=True)[0])

    try:
        log_thread.start()
        agent.train(num_processes=10, episodes=1e5)
        agent.save("model")
        print("FINISHED TRAINING")
        # agent.evaluate(10)
        # stop logger
        queue.put(None)
        log_thread.join()

    except KeyboardInterrupt as e:
        # if interrupt collect thread first
        queue.put(None)
        agent.save("model2")
        log_thread.join()
        raise e


if __name__ == "__main__":
    main()
