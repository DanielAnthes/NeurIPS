# A3C

- "Asynchronous Advantage Actor Critic"
    - Asynchronous:
	- play many games in parallel and "collaboratively" learn to solve the problem
    - Actor:
	- the agent learns to "act" in the environment and maintains a **policy** 
	- policy = for every state, estimate the "goodness" for every action
    - Critic:
	- the agent evaluates for every state what the expected future reward is
	- the "goodness" of the state it is in
    - Advantage:
	- the agent scales its learning experience with the magnitude of "estimation error"
	- advantage is how much the actually gained reward differs from the expected reward based on the agents predictions


# Reinforcement Learning
    
- in Reinforcement Learning an agent learns to solve a task in an environment by interacting with it

## Agent

- perceives the environment
    - (parameters, pixel values, ...)
- performs actions in the environment
- different types of agents:
    - model based
    - model free
	- actor
	- critic

## Environment

- the environment is "where the agent lives"
- reacts to input from the agent
- example: a game
    - physics
    - other agents
    - "rules of the game"
- deterministic vs stochastic

## Training

- the agent interacts with the environment
- often for many episodes / rollouts
- after every step the agent takes, it receives information about the new state of the environment and the reward or penalty incurred by the last action
- agent uses rewards to "reinforce" positive experiences and learn to avoid those that result in a penalty
- over time the agent improves

# References

Asynchronous Methods for Deep Reinforcement Learning, Mnih et al
