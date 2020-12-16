import numpy as np
import random
from typing import Iterable
import torch
from dqn import DQN, dqn_loss
import matplotlib.pyplot as plt 
import os
def set_random_seed(environment, seed):
    environment.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # NEW
    random.seed(seed)
    
def load_model(model_type, path_weights, environment, memory_size=20, model_kwargs={}):
    model = model_type(environment.observation_space.shape,
                                             environment.action_space.n,
                                             memory_size=memory_size,
                                              **model_kwargs)
    m = DQN(
        list(range(environment.action_space.n)),
        model,
        optimizer="sgd",
        loss_function=dqn_loss,
    )
    m.load_weights(path_weights)
    return m

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