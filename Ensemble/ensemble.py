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
sys.path.append("..")
from RNN.short_memory_models import *
import itertools

class Average_ensemble():
    def __init__(self, actions, models, normalize_preds = False, *args, **kwargs):
        self.actions = actions
        self.models = models
        self.n_models = len(self.models)
        self.n_actions = len(self.actions)
        self.normalize_preds = normalize_preds
        
    def get_action(self, state):
        pred_actions = np.zeros((self.n_models, self.n_actions))
        for i in range(self.n_models):
            a = self.models[i].predict(state[np.newaxis])
            if self.normalize_preds:
                a = a / np.linalg.norm(a, axis=-1)
            pred_actions[i] = a
            
        avg_actions = np.mean(pred_actions, axis=0)
        #print(pred_actions.shape)
        #print(avg_actions)
        return np.argmax(avg_actions)
    
    
def evaluate_memory_models(model, environment, memory_size=20, test_episodes=100):
    steps_done = 0
    episodes_done = 0
    R_episodes = []
    verbose_interval = 10
    render_interval = 10
    for _ in range(test_episodes):
        frame = environment.reset()
        state = np.zeros((memory_size, *environment.observation_space.shape))
        state[-1] = frame

        terminal = False
        R_episode: float = 0.0
        while not terminal:
            a = model.get_action(state)
            next_frame, r, terminal, _ = environment.step(a)

            next_state = np.vstack([np.delete(deepcopy(state), obj=0, axis=0), next_frame])
            state = next_state
            steps_done += 1

            R_episode += r
        episodes_done += 1
        R_episodes.append(R_episode)
        if episodes_done % verbose_interval == 0:
            print(f"episode: {episodes_done}, R: {R_episode:.2f},"
                  f" R_mean: {np.mean(R_episodes):.2f}", np.std(R_episodes))
        if episodes_done % render_interval == 0 and episodes_done > 0:
            show_rewards(R_episodes, block=False)
    return R_episodes

def load_model(model_type, path_weights, environment):
    model = model_type(environment.observation_space.shape[0],
                                             environment.action_space.n,
                                             memory_size=20)
    m = DQN(
        list(range(env.action_space.n)),
        model,
        optimizer="sgd",
        loss_function=dqn_loss,
    )
    m.load_weights(path_weights)
    return m


if __name__ == "__main__":
    seed=42
    models = [
            {"title": "Short memory LSTM Rewards", "model_type": SMLSTM, "memory_size": 20, "model_name": "SMLSTM"},
            {"title": "Short memory GRU Rewards", "model_type": SMGRU, "memory_size": 20, "model_name": "SMGRU"},
            {"title": "Prototypical short memory RNN Rewards", "model_type": PotoSMRNN, "memory_size": 20, "model_name": "PotoSMRNN"},
        ]
    envs = ["LunarLander-v2", "Acrobot-v1", "CartPole-v1"]

    for _env in envs:
        env = gym.make(_env)
        set_random_seed(env, seed)
        for l in range(1,len(models)):
            for ensemble in itertools.combinations(models, l):
                list_models = []
                for m in ensemble:
                    print(m["model_name"], end=" ")
                    weights_path = "trained_models"+m["model_name"] + "_" + _env + ".weights"
                    list_models.append(load_model(m["model_type"], weights_path, env))

                actions = list(range(env.action_space.n))
                avg_ensemble = Average_ensemble(actions, list_models, normalize_preds=True)
                history = evaluate_memory_models(avg_ensemble, env, test_episodes=100)
                np.save("test_results/"+'_'.join([m["model_name"] for m in ensemble])+_env+".npy", history)
                print()