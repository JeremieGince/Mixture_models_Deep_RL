import os
import random
import time
from copy import deepcopy
from typing import Tuple, List, Union, Iterable

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from poutyne import Model

from Models.short_memory_model import SMModel

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size = buffer_size

        # List[(s, a, r, next_s, episode_done)]
        self.data: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def store(self, element: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        """
        if len(self.data) >= self.__buffer_size:
            self.data.pop(0)

        self.data.append(element)

    def get_batch(self,
                  batch_size: int
                  ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """
        Returns a list of batch_size elements from the buffer.
        """
        return random.choices(self.data, k=batch_size)


class DQN(Model):
    def __init__(self, actions, *args, **kwargs):
        self.actions = actions
        super().__init__(*args, **kwargs)

    def get_action(self,
                   state: Union[torch.Tensor, np.ndarray],
                   epsilon: float) -> int:
        """
        Returns the selected action according to an epsilon-greedy policy.
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.predict([
                state[np.newaxis,],
            ], batch_size=1).squeeze()).item()

    def soft_update(self, other, tau):
        """
        Code for the soft update between a target network (self) and
        a source network (other).

        The weights are updated according to the rule in the assignment.
        """
        new_weights = {}

        own_weights = self.get_weight_copies()
        other_weights = other.get_weight_copies()

        for k in own_weights:
            new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

        self.set_weights(new_weights)


def format_batch(
        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
        target_network: DQN,
        gamma: float
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Input :
        - batch, a list of n=batch_size elements from the replay buffer
        - target_network, the target network to compute the one-step lookahead target
        - gamma, the discount factor

    Returns :
        - states, a numpy array of size (batch_size, state_dim) containing the states in the batch
        - (actions, targets) : where actions and targets both
                      have the shape (batch_size, ). Actions are the
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.
    """
    # TODO: Implement
    states = np.array([e[0] for e in batch])
    actions = np.array([e[1] for e in batch])

    next_states = np.array([e[3] for e in batch])
    target_predictions = target_network.predict([next_states, ], batch_size=len(next_states))
    targets = np.array([e[2] + gamma * np.max(q) * (1 - e[4]) for e, q in zip(batch, target_predictions)])
    return (states,), (actions, targets)


def dqn_loss(y_pred: torch.Tensor, y_target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Input :
        - y_pred, (batch_size, n_actions) Tensor outputted by the network
        - y_target = (actions, targets), where actions and targets both
                      have the shape (batch_size, ). Actions are the
                      selected actions according to the target network
                      and targets are the one-step lookahead targets.

    Returns :
        - The DQN loss
    """
    # TODO: Implement
    actions, targets = y_target
    return torch.mean(torch.pow(targets.detach() - y_pred[np.arange(y_pred.shape[0]), actions.long()], 2))


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # NEW
    random.seed(seed)


def show_rewards(R: Iterable, **kwargs):
    plt.plot(R)
    plt.yticks(np.arange(max(np.min(R), -200), np.max(R) + 1, 50))
    plt.grid()
    title = kwargs.get("title", "Reward per episodes")
    plt.title(title)
    plt.ylabel("Reward [-]")
    plt.xlabel("Episodes [-]")

    subfolder = kwargs.get("subfolder", False)
    if subfolder:
        os.makedirs(f"figures/{subfolder}/", exist_ok=True)
        plt.savefig(f"figures/{subfolder}/Projet_{title.replace(' ', '_').replace(':', '_')}.png", dpi=300)
    else:
        os.makedirs("RNN/figures/", exist_ok=True)
        plt.savefig(f"figures/Projet_{title.replace(' ', '_').replace(':', '_')}.png", dpi=300)
    plt.show(block=kwargs.get("block", True))


def main(
        batch_size: int,
        gamma: float,
        buffer_size: int,
        seed: int,
        tau: float,
        training_interval: int,
        lr: float,
        epsilon_decay: float,
        min_epsilon: float,
        model_kwargs: dict = None,
        **kwargs
):
    if model_kwargs is None:
        model_kwargs = {}
    #save_best_mean = kwargs.get("save_best_mean", 0)
    env_name = kwargs.get("env", "LunarLander-v2")
    environment = gym.make(env_name)
    set_random_seed(environment, seed)

    print(f"environment.observation_space.shape: {environment.observation_space.shape}")
    actions = list(range(environment.action_space.n))
    model: SMModel = kwargs.get("model_type", SMLSTM)(environment.observation_space.shape,
                                                      environment.action_space.n,
                                                      memory_size=kwargs.get("memory_size", 20),
                                                      **model_kwargs)
    policy_net = DQN(
        actions,
        model,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        loss_function=dqn_loss,
    )
    # NEW: pass a deep copy of the model
    target_net = DQN(actions, deepcopy(model), optimizer="sgd", loss_function=dqn_loss, )
    replay_buffer = ReplayBuffer(buffer_size)

    model.to(DEVICE)
    policy_net.to(DEVICE)

    training_done = False
    max_episode = kwargs.get("max_episode", 600)
    episodes_done = 0
    steps_done = 0
    epsilon = 1.0
    verbose_interval = kwargs.get("verbose_interval", 100)
    render_interval = kwargs.get("render_interval", verbose_interval)

    R_episodes = []
    start_time = time.time()
    best_score = -np.inf
    while not training_done:
        frame = environment.reset()
        state = np.zeros((model.memory_size, *environment.observation_space.shape))
        state[-1] = frame

        terminal = False
        R_episode: float = 0.0
        while not terminal:
            a = policy_net.get_action(state, epsilon)
            next_frame, r, terminal, _ = environment.step(a)

            next_state = np.vstack([np.delete(deepcopy(state), obj=0, axis=0), next_frame])

            replay_buffer.store((state, a, r, next_state, terminal))
            state = next_state
            steps_done += 1

            R_episode += r

            if steps_done % training_interval == 0:
                if len(replay_buffer.data) >= batch_size:
                    batch = replay_buffer.get_batch(batch_size)
                    x, y = format_batch(batch, target_net, gamma)
                    loss = policy_net.train_on_batch(x, y)
                    target_net.soft_update(policy_net, tau)

            if episodes_done % render_interval == 0 and episodes_done > 0:
                environment.render()

        R_episodes.append(R_episode)
        if episodes_done % verbose_interval == 0:
            if episodes_done == 0:
                print(f"episode: {episodes_done}, R: {R_episode:.2f}, epsilon: {epsilon:.2f}")
            else:
                print(f"episode: {episodes_done}, R: {R_episode:.2f},"
                      f" R_mean_100: {np.mean(R_episodes[-100:]):.2f}, epsilon: {epsilon:.2f}")
        if episodes_done % render_interval == 0 and episodes_done > 0:
            show_rewards(R_episodes, block=False,
                         title=kwargs.get("title", "RNN Rewards") + f", epi {episodes_done} " + f" env: {env_name}",
                         subfolder=f"temp/{env_name}")
        
        if np.mean(R_episodes[-100:]) > best_score:
            best_score = np.mean(R_episodes[-100:])
            policy_net.save_weights(kwargs.get("filename_weights", "model_weights")+".weights")
        
        if episodes_done >= max_episode:
            training_done = True
        epsilon = max(min_epsilon, epsilon_decay * epsilon)
        episodes_done += 1

    show_rewards(R_episodes, block=True,
                 title=kwargs.get("title", "RNN Rewards") + f" env: {env_name}", subfolder=f"{env_name}")
    print(f"\n episodes: {episodes_done},"
          f" R_mean_100: {np.mean(R_episodes[-100:]):.2f},"
          f"Elapse time: {time.time() - start_time:.2f} [s] \n")
    environment.close()


def device_setup():
    import torch
    import sys
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    from subprocess import call
    call(["nvcc", "--version"])
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'-' * 25}\nDEVICE: {device}\n{'-' * 25}\n")

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


if __name__ == "__main__":
    from Models.cnn import SMCNN
    from Models.fully_connected import SMNNModel
    from Models.gru import SMGRU
    from Models.lstm import SMLSTM
    from Models.prototypical_rnn import PotoSMRNN
    # from Models.quantum_fully_connected import SMQNNModel

    device_setup()

    models = [
        {"title": "Short memory LSTM Rewards", "model_type": SMLSTM, "memory_size": 20,
         "model_kwargs": {"name": "LSTM"}, },
        {"title": "Short memory GRU Rewards", "model_type": SMGRU, "memory_size": 20,
         "model_kwargs": {"name": "GRU"}, },
        {"title": "Prototypical short memory RNN Rewards", "model_type": PotoSMRNN, "memory_size": 20,
         "model_kwargs": {"name": "RNN"}, },
        {"title": "Short memory space CNN Rewards", "model_type": SMCNN, "memory_size": 20,
         "model_kwargs": {"permute": False, "name": "space_CNN"}, },
        {"title": "Short memory temporal CNN Rewards", "model_type": SMCNN, "memory_size": 20,
         "model_kwargs": {"permute": True, "name": "temporal_CNN"},
         },
        {"title": "Short memory NN Rewards", "model_type": SMNNModel, "memory_size": 20,
         "model_kwargs": {"name": "NN"},
         },
        # {"title": "Short memory Quantum NN Rewards", "model_type": SMQNNModel, "memory_size": 20,
        #  "model_kwargs": {"nb_qubits": 2, "nb_q_layer": 2, "hidden_dim": 2, "n_hidden_layers": 2},
        #  },
    ]
    envs = [
        "LunarLander-v2",
        "Acrobot-v1",
        "CartPole-v1",
        # "Breakout-v0",
    ]

    for _env in envs:
        for _m in models:
            print(f"\n{'-' * 75}\n\tenv: {_env}\n\ttitle: {_m['title']}\n{'-' * 75}\n")
            main(
                batch_size=64,
                gamma=0.99,
                buffer_size=int(1e5),
                seed=42,
                tau=1e-2,
                training_interval=_m["memory_size"]//2,
                lr=1e-3,
                epsilon_decay=0.995,
                min_epsilon=0.01,
                verbose_interval=100,
                render_interval=10000,
                max_episode=1_000,
                title=_m["title"],
                model_type=_m["model_type"],
                memory_size=_m["memory_size"],
                model_kwargs=_m["model_kwargs"],
                env=_env,
                filename_weights="trained_models/"+_m["model_kwargs"]["name"]+"_"+_env
            )
