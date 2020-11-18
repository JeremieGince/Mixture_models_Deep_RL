import numpy as np
import gym

from matplotlib import pyplot as plt
from tilecoding import TileCoder


def get_tile_coder(environment):
    return TileCoder(
        environment.observation_space.high,
        environment.observation_space.low,
        num_tilings=8,
        tiling_dim=8,
        max_size=4096,
    )


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)


def plot_reward(episodes_reward, lambda_):
    plt.plot(episodes_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Lambda: {lambda_}")
    plt.show()


def print_average_last_100_episodes(lambda_, episodes_reward):
    if len(episodes_reward) >= 100:
        last_100_episodes_reward = np.array(episodes_reward[-100:])
        average_reward = np.mean(last_100_episodes_reward)
        print(f"Lambda: {lambda_}; Average last 100 episodes: {average_reward}")


def find_action(state, tile_coder, nb_actions, w):
    actions_values = []
    actions = list(range(nb_actions))
    for action in actions:
        action_value = np.sum(w[tile_coder.phi(state, action)])
        actions_values.append(action_value)
    return np.argmax(actions_values)


def sarsa_lambda(env, nb_episodes=500, alpha=0.1, lambda_=0.9, gamma=1.0, render=False, plot=False):
    nb_actions = env.action_space.n
    tile_coder = get_tile_coder(env)

    w = np.zeros(tile_coder.size)
    #z = np.zeros(tile_coder.size)
    episodes_reward = []
    resolved = False
    for i_episode in range(nb_episodes):
        z = np.zeros(tile_coder.size)
        state = env.reset()
        total_reward = 0
        done = False
        t = 0
        action = find_action(state, tile_coder, nb_actions, w)
        while not done:
            if render:
                env.render()

            t += 1
            next_state, reward, done, _ = env.step(action)

            tiles = tile_coder.phi(state, action)
            delta = reward - np.sum(w[tiles])
            z[tiles] = 1
            if done:
                w += alpha * delta * z
            else:
                next_action = find_action(next_state, tile_coder, nb_actions, w)
                tiles_next_state = tile_coder.phi(next_state, next_action)
                delta += gamma * np.sum(w[tiles_next_state])
                w += alpha * delta * z
                z *= gamma * lambda_
                action = next_action
                state = next_state

            total_reward += reward

        # print(f"Episode {i_episode + 1}: {total_reward}")
        episodes_reward.append(total_reward)
        if len(episodes_reward) >= 100 and not resolved:
            last_100_episodes_reward = np.array(episodes_reward[-100:])
            average_reward = np.mean(last_100_episodes_reward)
            if average_reward > -110:
                print(f"Resolved after {i_episode + 1} episodes; Average over last 100 episodes: {average_reward}")
                resolved = True

    if plot:
        plot_reward(episodes_reward, lambda_)

    print_average_last_100_episodes(lambda_, episodes_reward)
    #print(f"Nombre d'épisodes: {i_episode + 1}")


if __name__ == "__main__":
    seed = 42
    environment = gym.make("MountainCar-v0")
    set_random_seed(environment, seed)
    #sarsa_lambda(environment, plot=False)

    seeds = list(range(50))
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    #lambdas = [0, 0.9, 0.95, 0.99, 1.0]
    #lambdas = []
    for lambda_ in lambdas:
        environment = gym.make("MountainCar-v0")
        set_random_seed(environment, seed)
        print(f"Pour lambda: {lambda_}")
        sarsa_lambda(environment, lambda_=lambda_, plot=True)
        print("")
