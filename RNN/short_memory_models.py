import os
import random
import time
from copy import deepcopy  # NEW
from typing import Tuple, List, Union, Iterable

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from poutyne import Model
from torch import nn

from RNN.ConvNd import ConvNd

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


class SMModel(torch.nn.Module):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__()
        self.in_dim = in_shape
        self.out_dim = out_shape
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.memory_size: int = memory_size
        self.kwargs = kwargs
        self.c0 = torch.zeros(size=(1, hidden_dim,))
        self.h0 = torch.zeros(size=(1, hidden_dim,))

    def forward(self, *inputs):
        raise NotImplementedError()


class SMLSTM(SMModel):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )
        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim)

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(memory_size * hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        state_features = self.backbone(state.float())

        # lstm_output, _ = self.lstm(state_features.permute(1, 0, 2), (self.h0, self.c0))
        lstm_output, _ = self.lstm(state_features.permute(1, 0, 2))
        env_features = torch.flatten(lstm_output.permute(1, 0, 2), start_dim=1)
        q_values: torch.Tensor = self.q_predictor(env_features)
        return q_values


class PotoSMRNN(SMModel):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )

        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.state_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.context_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.fusion_layer = torch.nn.Sequential(*[
            self.linear_block(2 * hidden_dim, hidden_dim)
        ])

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        curr_state = state[:, -1]
        proto_context = torch.mean(state[:, :-1], dim=1)

        state_features = self.state_backbone(curr_state.float())
        context_features = self.context_backbone(proto_context.float())
        fusion_state = torch.cat([state_features, context_features], dim=-1)
        fusion_features = self.fusion_layer(fusion_state)
        q_values = self.q_predictor(fusion_features)
        return q_values


class SMGRU(SMModel):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )
        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.gru = torch.nn.GRU(hidden_dim, hidden_dim)

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(memory_size * hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        state_features = self.backbone(state.float())

        # lstm_output, _ = self.lstm(state_features.permute(1, 0, 2), (self.h0, self.c0))
        lstm_output, _ = self.gru(state_features.permute(1, 0, 2))
        env_features = torch.flatten(lstm_output.permute(1, 0, 2), start_dim=1)
        q_values = self.q_predictor(env_features)
        return q_values


class SMCNN(SMModel):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 1,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        n_dim = len(in_shape)
        self.n_dim = n_dim
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )
        stride = 1
        self.permute = kwargs.get("permute", False)

        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.conv1d_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            nn.Conv1d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=stride, bias=True),
            nn.BatchNorm1d(c_o),
            nn.ReLU(),
        )
        self.conv2d_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            nn.Conv2d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=stride, bias=True),
            nn.BatchNorm2d(c_o),
            nn.ReLU(),
        )
        self.conv3d_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            nn.Conv3d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=stride, bias=True),
            nn.BatchNorm3d(c_o),
            nn.ReLU(),
        )
        self.convNd_bloc = lambda c_i, c_o, k: torch.nn.Sequential(
            ConvNd(in_channels=c_i, out_channels=c_o, num_dims=n_dim,
                   kernel_size=k, stride=stride, padding=0, use_bias=True),
            nn.SyncBatchNorm(c_o),
            nn.ReLU(),
        )

        self.conv_func_ndim = {
            1: self.conv1d_bloc,
            2: self.conv2d_bloc,
            3: self.conv3d_bloc,
        }

        self.conv_bloc_gen = self.conv_func_ndim[n_dim] if n_dim in self.conv_func_ndim else self.convNd_bloc

        if self.permute:
            self.complete_in_shape = (1, *in_shape, memory_size)

            self.sh_idx = list(range(len(self.complete_in_shape)))
            c = self.sh_idx.pop(1)
            self.sh_idx.append(c)
        else:
            self.complete_in_shape = (1, memory_size, *in_shape)

        k_list = list(reversed(range(0, max(0, self.complete_in_shape[-1] // 2))))
        layers = [
            self.conv_bloc_gen(self.complete_in_shape[1], hidden_dim, k_list[0]),
        ]
        for i, k in enumerate(k_list):
            if k > 0 and i < n_hidden_layers:
                layers.append(self.conv_bloc_gen(hidden_dim, hidden_dim, k))

        self.backbone = torch.nn.Sequential(*layers)

        # _in = torch.ones(self.complete_in_shape)
        # for layer in layers:
        #     print(f"_in.shape: {_in.shape}")
        #     print(f"layer: {layer}")
        #     _out = layer(_in)
        #     print(f"_out.shape: {_out.shape}")
        #     _in = _out

        self.backbone_output_shape: tuple = tuple(self.backbone(torch.ones(self.complete_in_shape)).shape)
        hh_dim: int = int(np.prod(self.backbone_output_shape))
        self.q_predictor = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(hh_dim, hh_dim),
            torch.nn.Linear(hh_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        if self.permute:
            state = state.permute(self.sh_idx)
        state_features = self.backbone(state.float())
        q_values = self.q_predictor(state_features)
        return q_values


class SMNNModel(SMModel):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )

        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(memory_size*np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.q_predictor = torch.nn.Sequential(*[
            torch.nn.Linear(hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        features = self.backbone(state.float())
        q_values = self.q_predictor(features)
        return q_values


class SMQNNModel(SMModel):
    """
    Neural Network with 3 hidden layers of hidden dimension 64.
    """

    def __init__(self,
                 in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 2,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
        super().__init__(
            in_shape,
            out_shape,
            n_hidden_layers,
            hidden_dim,
            memory_size,
            **kwargs,
        )
        from Modules.classifiers.layers import QuantumPseudoLinearLayer
        self.complete_in_shape = (1, memory_size, *in_shape)
        self.nb_qubits = kwargs.get("nb_qubits", 2)
        kwargs["nb_qubits"] = self.nb_qubits
        self.linear_block = lambda i, o: torch.nn.Sequential(*[
            torch.nn.Linear(i, o),
            torch.nn.ReLU(),
        ])

        self.q_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(memory_size*np.prod(in_shape), self.nb_qubits),
            *[QuantumPseudoLinearLayer(**kwargs)
              for _ in range(kwargs.get("nb_q_layer", 2))],
        ])
        self.q_backbone_output_shape: tuple = tuple(self.q_backbone(torch.ones(self.complete_in_shape)).shape)
        q_hh_dim: int = int(np.prod(self.q_backbone_output_shape))

        self.c_backbone = torch.nn.Sequential(*[
            nn.Flatten(),
            self.linear_block(memory_size*np.prod(in_shape), hidden_dim),
            *[self.linear_block(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        ])

        self.fusion_layer = torch.nn.Sequential(*[
            self.linear_block(hidden_dim+q_hh_dim, hidden_dim)
        ])

        self.q_predictor = torch.nn.Sequential(*[
            nn.Flatten(),
            torch.nn.Linear(hidden_dim, int(np.prod(out_shape)))
        ])

    def forward(self, *inputs):
        [state, ] = inputs
        q_features = self.q_backbone(state.float())
        c_features = self.c_backbone(state.float())
        fusion_features = torch.cat([q_features, c_features], dim=-1)
        fusion_features = self.fusion_layer(fusion_features)
        q_values = self.q_predictor(fusion_features)
        return q_values


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
        os.makedirs("figures/", exist_ok=True)
        plt.savefig(f"figures/Projet_{title.replace(' ', '_').replace(':', '_')}.png", dpi=300)
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
        model_kwargs: dict = {},
        **kwargs
):
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
                      f" R_mean_100: {np.mean(R_episodes[:-100]):.2f}, epsilon: {epsilon:.2f}")
        if episodes_done % render_interval == 0 and episodes_done > 0:
            show_rewards(R_episodes, block=False,
                         title=kwargs.get("title", "RNN Rewards") + f", epi {episodes_done} " + f" env: {env_name}",
                         subfolder=f"temp/{env_name}")
        if episodes_done >= max_episode:
            training_done = True
        epsilon = max(min_epsilon, epsilon_decay * epsilon)
        episodes_done += 1

    show_rewards(R_episodes, block=True,
                 title=kwargs.get("title", "RNN Rewards") + f" env: {env_name}", subfolder=f"{env_name}")
    print(f"\n episodes: {episodes_done},"
          f" R_mean_100: {np.mean(R_episodes[:-100]):.2f},"
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
    print(f"\n{'-' * 25}\nDEVICE: {device}\n{'-' * 25}\n")

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
    device_setup()

    models = [
        # {"title": "Short memory LSTM Rewards", "model_type": SMLSTM, "memory_size": 20,
        #  "model_kwargs": {}, },
        # {"title": "Short memory GRU Rewards", "model_type": SMGRU, "memory_size": 20,
        #  "model_kwargs": {}, },
        # {"title": "Prototypical short memory RNN Rewards", "model_type": PotoSMRNN, "memory_size": 20,
        #  "model_kwargs": {}, },
        {"title": "Short memory space CNN Rewards", "model_type": SMCNN, "memory_size": 20,
         "model_kwargs": {"permute": False}, },
        {"title": "Short memory temporal CNN Rewards", "model_type": SMCNN, "memory_size": 20,
         "model_kwargs": {"permute": True},
         },
        {"title": "Short memory NN Rewards", "model_type": SMNNModel, "memory_size": 20,
         "model_kwargs": {},
         },
        # {"title": "Short memory Quantum NN Rewards", "model_type": SMQNNModel, "memory_size": 20,
        #  "model_kwargs": {"nb_qubits": 2, "nb_q_layer": 2, "hidden_dim": 64, "n_hidden_layers": 2},
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
                render_interval=100,
                max_episode=800,
                title=_m["title"],
                model_type=_m["model_type"],
                memory_size=_m["memory_size"],
                model_kwargs=_m["model_kwargs"],
                env=_env,
            )
