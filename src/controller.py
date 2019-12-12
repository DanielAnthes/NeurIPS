# from neurosmash import NeurosmashEnvironment
import sys
import curses

from neurosmash_environment import NeurosmashEnvironment as NSenv

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


def connect(timescale=1, size=768, port=13000, ip="127.0.0.1"):
    return NSenv(timescale, size, port, ip)

def run(env, screen):
    memory = []
    while True:
        _in = get_input(screen)
        if _in == -1:
            break
        if _in != '':
            screen.addstr(0,0,str(_in))
            env.step(_in)

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
    try:
        env = connect(timescale=1, size=768, port=13000, ip="127.0.0.1")
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
