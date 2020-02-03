"""Runs a random agent in a multiprocessed fashion to generate a multitude of states and save them for further processing"""
import os
from multiprocessing import Manager, Process

from neurips2019.agents.NeuroRandomAgent import NeuroRandomAgent as RandAgent
from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment as NSenv
from neurips2019.preprocessing.neurosmash_state_processing import save_states


def worker(_id, epochs, port, save_dir):
    path = os.path.join(save_dir, f"w{_id:02d}")
    if not os.path.isdir(path):
        os.makedirs(path)
    print(f"Worker {_id:02d}: Starting @ port {port} for {epochs} epochs")
    try:
        env = NSenv(size=64, timescale=3, port=port)
        agent = RandAgent(env, True, save_dir)
    except ConnectionRefusedError:
        print(f"Worker {_id:02d}: Error when connecting to Environment. Aborting.")
        return

    for e in range(epochs):
        e = e + 1
        start_state = e // 10
        agent.evaluate(**{"save.start_state": start_state})
        if e % 50 == 0:
            print(f"Worker {_id:02d}: Finished Epoch {e}. Saving Buffer.")
            save_states(agent.states, f"randomagent-{e//50}", path)
            agent.states = []

    if len(agent.states) > 0:
        save_states(agent.states, f"randomagent-{e//50 + 1}", path)

    print(f"Worker {_id:02d}: Done")


def main(n_proc, total_epochs):
    save_dir = os.path.join("logs", "NeuroSmashStates", "run01")
    epochs = total_epochs // n_proc

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    port_start = 8000
    processes = []
    print("Starting...")
    for i in range(n_proc):
        p = port_start+i
        p = Process(target=worker, args=(i+1, epochs, p, save_dir))
        p.start()
        processes.append(p)

    # done_count = 0
    # while True:
    #     states = queue.get()
    #     if type(states) is str and states == "done":
    #         done_count += 1
    #         if done_count == n_proc:
    #             print(f"Done Count: {done_count}")
    #             break
    #     elif type(states) == list:
    #         save_states(states, "randomagent", save_dir)

    print("Joining....")
    for p in processes:
        p.join()


if __name__ == "__main__":
    main(
        n_proc=12,
        total_epochs=24000
    )