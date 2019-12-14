"""
A keyboard controller for neurosmash. ENV variables are up top below imports.

For curses to work on windows, third party binaries need to be installed.
"""
# from neurosmash import NeurosmashEnvironment
import sys
import curses
import time

from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment as NSenv


ENV_TIMESCALE=5
ENV_SIZE=768
ENV_PORT=13000
ENV_IP="127.0.0.1"


def get_input(screen):
    char = screen.getch()
    if char == ord('q'):
        return -1
    elif char == curses.KEY_RIGHT:
        return 2
    elif char == curses.KEY_LEFT:
        return 1
    elif char == curses.KEY_DOWN:
        return 0
    elif char == curses.KEY_BACKSPACE:
        return 4
    else:
        return 0

def connect():
    return NSenv(ENV_TIMESCALE, ENV_SIZE, ENV_PORT, ENV_IP)

def run(env, screen):
    memory = []
    state = rew = done = nstate = None
    while True:
        _in = get_input(screen)
        if _in == -1:
            break
        if _in == '': _in = 0
        if _in == 4: 
            state = env.reset()
            rew = 0
            done = False
            continue
        screen.addstr(1,0,str(_in))
        nstate, rew, done = env.step(_in)
        time.sleep(0.1 / ENV_TIMESCALE)
        memory.append((state, rew, done, nstate))
        state = nstate

    return memory

def save(result):
    pass

def main():
    # get the curses screen window
    screen = curses.initscr()
    # turn off input echoing
    curses.noecho()
    # respond to keys immediately (don't wait for enter)
    curses.cbreak()
    # map arrow keys to special values
    screen.keypad(True)
    #
    screen.addstr(0,0,"Left / Right Arrow to turn. Lower Arrow for straight. q to quit.\n")
    try:
        env = connect()
        result = run(env, screen)
        save(result)
    finally:
        # shut down cleanly
        curses.nocbreak()
        screen.keypad(0)
        curses.echo()
        curses.endwin()


if __name__ == "__main__":
    main()
