from DQN import *

# Now we run the REINFORCE agent within the environment. Note that we update the agent after each episode for simplicity.
# First, we should restart the server from the GUI

environment = Environment()
network = MLP(n_output=2, n_hidden=3)
agent = REINFORCEAgent(network, optimizer=Adam())

episode_count = 1000


R = np.zeros(episode_count)


for i in tqdm.trange(episode_count):

    reward, state, status = environment.reset()

    loss = 0
    while True:

        action, policy = agent.step(reward, state)

        reward, state, status = environment.step(action[0])

        # get reward associated with taking the previous action in the previous state
        agent.rewards.append(reward)
        R[i] += reward

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        agent.scores.append(agent.compute_score(action, policy))

        # we learn at the end of each episode
        if status[0] == 1:

            loss += agent.compute_loss()

            agent.model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            agent.optimizer.update()

            break
