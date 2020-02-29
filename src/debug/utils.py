import cv2

def resize(state, size):
        state = cv2.resize(state, size, interpolation=cv2.INTER_CUBIC) # resize to cnn input size
        state = state.reshape(*size,1)
        state = state.transpose((2, 0, 1)).copy() # convert to pytorch format
        return state

def get_state(env):
    state = env.render(mode="rgb_array")
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = resize(state, (64,64))
    return state
