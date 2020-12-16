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
from utils import set_random_seed, show_rewards, load_model
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import itertools
from dqn import DQN, dqn_loss, ReplayBuffer, format_batch

from Ensemble.attn_ensemble import *



def main(
        ensemble,
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
    env_memory_size = max([m["memory_size"] for m in ensemble])
    list_models = []
    for m in ensemble:
        print(m["model_kwargs"]["name"])
        weights_path = "trained_models/"+m["model_kwargs"]["name"] + "_" + _env + ".weights"

        list_models.append(load_model(m["model_type"], weights_path, env, m["memory_size"], model_kwargs=m["model_kwargs"]))


    attn_fusion = Fusion_network(list_models,
                                 len(actions),
                                 environment.observation_space.shape[0],
                                 len(list_models))
    attn_ensemble = Attention_ensemble(actions,
                                       list_models,
                                       network=attn_fusion,
                    optimizer=torch.optim.Adam(attn_fusion.parameters(), lr=lr),
                    loss_function=dqn_loss)
    

    target_net = Attention_ensemble(actions, list_models,
                    network=deepcopy(attn_fusion),
                    optimizer="sgd",
                    loss_function=dqn_loss)
    replay_buffer = ReplayBuffer(buffer_size)

    
    
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
        state = np.zeros((env_memory_size, *environment.observation_space.shape))
        state[-1] = frame

        terminal = False
        R_episode: float = 0.0
        while not terminal:
            a = attn_ensemble.get_action(state, epsilon)
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
                    loss = attn_ensemble.train_on_batch(x, y)
                    target_net.soft_update(attn_ensemble, tau)

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
    attn_ensemble.save_weights(kwargs.get("filename_weights", "model_weights")+".weights")
    

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
    from Models.prototypical_rnn import ProtoSMRNN

    device_setup()

    models = [
        {"title": "Short memory LSTM Rewards", "model_type": SMLSTM, "memory_size": 20,
         "model_kwargs": {"name": "LSTM"}, },
        {"title": "Short memory GRU Rewards", "model_type": SMGRU, "memory_size": 20,
         "model_kwargs": {"name": "GRU"}, },
        {"title": "Prototypical short memory RNN Rewards", "model_type": ProtoSMRNN, "memory_size": 20,
         "model_kwargs": {"name": "RNN"}, },
        {"title": "Short memory space CNN Rewards", "model_type": SMCNN, "memory_size": 20,
         "model_kwargs": {"permute": False, "name": "space_CNN"}, },
        {"title": "Short memory NN Rewards", "model_type": SMNNModel, "memory_size": 1,
         "model_kwargs": {"name": "NN"},
         }
    ]
    envs = [
        "LunarLander-v2",
        "Acrobot-v1",
        "CartPole-v1",
        # "Breakout-v0",
    ]

    for _env in envs:
        env = gym.make(_env)
        print(f"\n{'-' * 75}\n\tenv: {_env}\n{'-' * 75}\n")
        for ensemble in itertools.combinations(models, len(models)):

            main(
                ensemble,
                batch_size=64,
                gamma=0.99,
                buffer_size=int(1e5),
                seed=42,
                tau=1e-2,
                training_interval=4,#_m["memory_size"]//2,
                lr=1e-3,
                epsilon_decay=0.99,
                min_epsilon=0.01,
                verbose_interval=100,
                render_interval=10000,
                max_episode=500,
                env=_env,
                filename_weights="trained_models/ensemble_attn_"+_env+".weights"
            )
