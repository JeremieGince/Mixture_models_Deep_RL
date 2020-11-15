import numpy as np
import gym
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


def find_action(state, tile_coder, nb_actions, w):
    actions_values = []
    actions = list(range(nb_actions))
    for action in actions:
        action_value = np.sum(w[tile_coder.phi(state, action)])
        actions_values.append(action_value)
    return np.argmax(actions_values)


def sarsa_lambda(env, nb_episodes=500, alpha=0.1, lambda_=0.9, gamma=1.0, render=False):
    nb_actions = env.action_space.n
    tile_coder = get_tile_coder(env)

    w = np.zeros(tile_coder.size)
    z = np.zeros(tile_coder.size)
    episodes_reward = []
    for i_episode in range(nb_episodes):
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

        print(f"Episode {i_episode + 1}: {total_reward}")
        episodes_reward.append(total_reward)
        if len(episodes_reward) >= 100:
            last_100_episodes_reward = np.array(episodes_reward[-100:])
            average_reward = np.mean(last_100_episodes_reward)
            if average_reward > -110:
                print(f"Average last 100 episodes: {average_reward}")
                break


if __name__ == "__main__":
    seed = 42
    environment = gym.make("MountainCar-v0")
    set_random_seed(environment, seed)
    sarsa_lambda(environment)
