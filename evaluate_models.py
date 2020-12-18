import random
from copy import deepcopy  # NEW
from typing import Tuple, List, Union

import gym
import numpy as np
import torch
from torch.autograd import Variable
from poutyne import Model
from torch.nn import functional as F
from typing import Tuple, List, Union, Iterable
import matplotlib.pyplot as plt
import os
import sys
import itertools
import time

from utils import set_random_seed, show_rewards, load_model

from dqn import DQN, dqn_loss


    
def evaluate_memory_models(model,
                           environment, 
                           memory_size=20, 
                           test_episodes=100,
                          verbose_interval = 100):
    steps_done = 0
    episodes_done = 0
    R_episodes = []
    
    render_interval = 10
    times = []
    for _ in range(test_episodes):
        frame = environment.reset()
        state = np.zeros((memory_size, *environment.observation_space.shape))
        state[-1] = frame

        terminal = False
        R_episode: float = 0.0
        while not terminal:
            t0 = time.time()
            a = model.get_action(state[-model.network.memory_size:], epsilon=0.0)
            times.append(time.time() - t0)
            next_frame, r, terminal, _ = environment.step(a)

            next_state = np.vstack([np.delete(deepcopy(state), obj=0, axis=0), next_frame])
            state = next_state
            steps_done += 1

            R_episode += r
        episodes_done += 1
        R_episodes.append(R_episode)
        if episodes_done % verbose_interval == 0:
            print(f"episode: {episodes_done}, R: {R_episode:.2f},"
                  f" R_mean: {np.mean(R_episodes):.2f}", f"std: {np.std(R_episodes):.2f}", f"{np.mean(times)} sec/pred")
        if episodes_done % render_interval == 0 and episodes_done > 0:
            show_rewards(R_episodes, block=False)
    return R_episodes


if __name__ == "__main__":
    from Models.cnn import SMCNN
    from Models.fully_connected import SMNNModel
    from Models.gru import SMGRU
    from Models.lstm import SMLSTM
    from Models.prototypical_rnn import ProtoSMRNN
    
    seed=42
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
         },
    ]
    
    envs = [
        "LunarLander-v2",
        "Acrobot-v1",
        "CartPole-v1",
        # "Breakout-v0",
    ]
    
    for _env in envs:
        env = gym.make(_env)
        set_random_seed(env, seed)
        for l in range(0,len(models)):
            model_use_memory = []
            m = models[l]
            print(_env, m["model_kwargs"]["name"])
            weights_path = "trained_models/"+m["model_kwargs"]["name"] + "_" + _env + ".weights"
            model_use_memory.append(m["memory_size"]>0)
            model = load_model(m["model_type"], weights_path, env, m["memory_size"], model_kwargs=m["model_kwargs"])
            actions = list(range(env.action_space.n))
            history = evaluate_memory_models(model, env, test_episodes=500)
            np.save("Experiments/test_results_models/"+m["model_kwargs"]["name"]+"_"+_env, history)
            print()