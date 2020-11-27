import random
from copy import deepcopy  # NEW
from typing import Tuple, List, Union, Iterable

import gym
import numpy as np
import torch
from poutyne import Model
from torch.nn import functional as F
from torch.autograd import Variable
import os

import matplotlib.pyplot as plt
import time

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.__buffer_size = buffer_size

        # List[(s, context, hidden, a, r, next_s, next_context, next_hidden, episode_done)]
        self.data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]] = []

    def store(self, element: Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]):
        """
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        """
        if len(self.data) >= self.__buffer_size:
            self.data.pop(0)

        self.data.append(element)

    def get_batch(self,
                  batch_size: int
                  ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]]:
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
                   context: Union[torch.Tensor, np.ndarray],
                   hidden: Union[torch.Tensor, np.ndarray],
                   epsilon: float) -> int:
        """
        Returns the selected action according to an epsilon-greedy policy.
        """
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.predict([
                state[np.newaxis, ],
                context[np.newaxis, ],
                hidden[np.newaxis, ]
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


class LSTMModel(torch.nn.Module):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 lstm_memory_size: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.backbone = torch.nn.Sequential(*[
            self.linear_block(in_dim, hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim)

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(lstm_memory_size*hidden_dim, out_dim)
        ])

        self.context_state = torch.zeros(size=(hidden_dim,))
        self.hidden_state = torch.zeros(size=(hidden_dim,))
        self.lstm_memory_size: int = lstm_memory_size

    def get_context_state(self) -> torch.Tensor:
        print(f"self.context_state: {self.context_state.shape}")
        return self.context_state

    def get_hidden_state(self) -> torch.Tensor:
        print(f"self.hidden_state: {self.hidden_state.shape}")
        return self.hidden_state

    def forward(self, *inputs):
        [state, context, hidden] = inputs
        # print(f"forward state: {state.shape}")
        #
        # print(f"forward context: {context.shape}")
        # print(f"forward hidden: {hidden.shape}")

        state_features = self.backbone(state.float())

        # print(f"forward state_features: {state_features.shape}")
        # print(f"forward state_features_permetued: {state_features.permute(1, 0, 2).shape}")

        lstm_output, (self.hidden_state, self.context_state) = self.lstm(state_features.permute(1, 0, 2),
                                                                         (hidden[np.newaxis, ], context[np.newaxis, ]))
        # print(f"lstm_output: {lstm_output.shape}")
        env_features = torch.flatten(lstm_output.permute(1, 0, 2), start_dim=1)
        # print(f"env_features: {env_features.shape}")
        q_values = self.q_predictor(env_features)
        # print(f"q_values: {q_values.shape}")
        return q_values


def format_batch(
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]],
        target_network: DQN,
        gamma: float
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
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
    contexts = np.array([e[1] for e in batch])
    hiddens = np.array([e[2] for e in batch])
    actions = np.array([e[3] for e in batch])

    next_states = np.array([e[5] for e in batch])
    next_contexts = np.array([e[6] for e in batch])
    next_hiddens = np.array([e[7] for e in batch])
    target_predictions = target_network.predict([next_states, next_contexts, next_hiddens], batch_size=len(next_states))
    targets = np.array([e[4] + gamma * np.max(q) * (1 - e[8]) for e, q in zip(batch, target_predictions)])
    return (states, contexts, hiddens), (actions, targets)


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
    print(f"y_pred: {y_pred.shape}, targets: {targets.shape}")
    return torch.mean(torch.pow(targets.detach() - y_pred[np.arange(y_pred.shape[0]), actions.long()], 2))


def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # NEW
    random.seed(seed)


def show_rewards(R: Iterable, **kwargs):
    plt.plot(R)
    plt.grid()
    title = kwargs.get("title", "Reward per episodes")
    plt.title(title)
    plt.ylabel("Reward [-]")
    plt.xlabel("Episodes [-]")
    os.makedirs("figures/", exist_ok=True)
    plt.savefig(f"figures/TP2_Q1_{title.replace(' ', '_')}.png", dpi=300)
    plt.show(block=kwargs.get("block", True))


# NEW : Added lr argument
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
        **kwargs
):
    environment = gym.make("LunarLander-v2")
    set_random_seed(environment, seed)

    actions = list(range(environment.action_space.n))
    model = LSTMModel(environment.observation_space.shape[0], environment.action_space.n)
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

    while not training_done:
        frame = environment.reset()
        state = np.zeros((model.lstm_memory_size, *environment.observation_space.shape))
        state[-1] = frame
        context = model.get_context_state().detach().numpy()
        hidden = model.get_hidden_state().detach().numpy()

        terminal = False
        R_episode: float = 0.0
        while not terminal:
            a = policy_net.get_action(state, context, hidden, epsilon)
            next_frame, r, terminal, _ = environment.step(a)

            next_state = np.vstack([np.delete(deepcopy(state), obj=0, axis=0), next_frame])
            next_context = model.get_context_state().detach().numpy()
            next_hidden = model.get_hidden_state().detach().numpy()

            print(f"next_state: {next_state.shape}")
            print(f"next_context: {next_context.shape}")
            print(f"next_hidden: {next_hidden.shape}")

            replay_buffer.store((state, context, hidden, a, r, next_state, next_context, next_hidden, terminal))
            state = next_state
            context = next_context
            hidden = next_hidden
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
            print(f"episode: {episodes_done}, R: {R_episode:.2f},"
                  f" R_mean: {np.mean(R_episodes):.2f}, epsilon: {epsilon:.2f}")
        if episodes_done % render_interval == 0 and episodes_done > 0:
            show_rewards(R_episodes, block=False)
        if episodes_done >= max_episode:
            training_done = True
        epsilon = max(min_epsilon, epsilon_decay * epsilon)
        episodes_done += 1

    show_rewards(R_episodes, block=True, title=f"LSTM Rewards")
    print(f"\n episodes: {episodes_done},"
          f" R_mean: {np.mean(R_episodes):.2f},"
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
    # print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    # print('Active CUDA Device: GPU', torch.cuda.current_device())

    # print('Available devices ', torch.cuda.device_count())
    # print('Current cuda device ', torch.cuda.current_device())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


if __name__ == "__main__":
    """
    All hyperparameter values and overall code structure are
    only given as a baseline. 

    You can use them if they help  you, but feel free to implement
    from scratch the required algorithms if you wish !
    """
    print(f"\n{'-' * 25}\nDEVICE: {DEVICE}\n{'-' * 25}\n")
    device_setup()

    main(
        batch_size=32,
        gamma=0.99,
        buffer_size=int(1e5),
        seed=42,
        tau=1e-2,
        training_interval=4,
        lr=1e-3,
        epsilon_decay=0.90,
        min_epsilon=0.01,
        verbose_interval=100,
        render_interval=1_000,
        max_episode=1_000,
    )
