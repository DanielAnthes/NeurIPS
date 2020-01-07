"""
A keyboard controller for neurosmash. ENV variables are up top below imports.

For curses to work on windows, third party binaries need to be installed.
"""
# from neurosmash import NeurosmashEnvironment
import sys
import curses
import time
from enum import Enum

from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment as NSenv

#############################
### Environment Variables ###
### Set them according to ###
### Neurosmash settings   ###
#############################
ENV_TIMESCALE=5
ENV_SIZE=768
ENV_PORT=13000
ENV_IP="127.0.0.1"

class KeyCodes(Enum):
    quit = -1
    noact = 0
    left = 1
    right = 2
    reset = 3

def get_input(screen):
    """
    Grab the characters from screen.
    """
    char = screen.getch()
    if char == ord('q'):
        return KeyCodes.quit
    elif char == ord('d'):
        return KeyCodes.right
    elif char == ord('a'):
        return KeyCodes.left
    elif char == ord('e'):
        return KeyCodes.reset
    else: # handle all other keys as no action
        return KeyCodes.noact

def connect():
    """Connect to Environment"""
    return NSenv(timescale=ENV_TIMESCALE, size=ENV_SIZE, ip=ENV_IP, port=ENV_PORT)

def run(env, screen):
    """
    Run the controller in a loop until Player quits.
    """
    memory = []
    state = rew = done = nstate = None
    while True:
        _in = get_input(screen)
        if _in == KeyCodes.quit:
            break
        # should technically not occur
        if _in == '':
            _in = KeyCodes.noact
        if _in == KeyCodes.reset: 
            state = env.reset()
            rew = 0
            done = False
            continue
            
        screen.addstr(1,0,str(_in.value))
        nstate, rew, done = env.step(_in.value)
        time.sleep(0.1 / ENV_TIMESCALE)
        memory.append((state, rew, done, nstate))
        state = nstate

    return memory

def save(result):
    """
    Save the episode in a format to enable imitation learning.
    
    To be implemented
    """
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
    # Add short info
    screen.addstr(0,0,"A & D to turn. S for straight. Q to quit. E to reset.\n")
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
