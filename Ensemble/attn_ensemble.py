import numpy as np
import torch
from poutyne import Model
from torch.nn import functional as F

class Fusion_network(torch.nn.Module):

    def __init__(self, models, n_actions, in_dim, out_dim, n_hidden_layers=3, hidden_dim=64):
        self.models = models
        self.n_models = len(models)
        self.n_actions = n_actions
        self.memory_size=1
        super().__init__()
        layers = [torch.nn.Linear(in_dim+self.n_models*self.n_actions, hidden_dim), torch.nn.ReLU()]
        #layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        #layers.append(torch.nn.Softmax())

        self.fa = torch.nn.Sequential(*layers)

    def get_actions(self, state):
        batch_size = state.shape[0]
        pred_actions = np.zeros((batch_size, self.n_models, self.n_actions))
        for i in range(self.n_models):
            a = self.models[i].predict(state[:, -self.models[i].network.memory_size:])


            #print(a.shape)
            pred_actions[:,i] = a
            pred_actions = pred_actions#/np.linalg.norm(pred_actions, axis=-1)[:,:,np.newaxis]
        return pred_actions    
        
    def format_states_actions(self, state, pred_actions):
        #print(state[:,-1].shape, pred_actions.reshape((pred_actions.shape[0],-1)).shape)
        #print(np.concatenate((pred_actions.reshape((pred_actions.shape[0],-1)), state[:,-1]), axis=-1).shape)
        return torch.cat((pred_actions.reshape((pred_actions.shape[0],-1)), state[:,-1]), dim=-1)
    
    def forward(self, state, return_weights=False):
        pred_actions = torch.from_numpy(self.get_actions(state))
        
        state_preds = self.format_states_actions(state, pred_actions)
        attn = self.fa(state_preds.float())#state.float()[:,-1])
        attn = torch.nn.functional.softmax(attn/1000, dim=-1)
        weighted_sum = torch.sum(attn.unsqueeze(-1)*pred_actions, axis=1)
        #print(attn[0])
        self.last_preds = weighted_sum
        self.last_attn = attn
        return weighted_sum
    
class Attention_ensemble(Model):
    def __init__(self, actions, models, weight_models=False, normalize_preds = False, *args, **kwargs):
        self.actions = actions
        self.models = models
        self.n_models = len(self.models)
        self.n_actions = len(self.actions)
        self.normalize_preds = normalize_preds
        self.weight_models = weight_models

        super().__init__(*args, **kwargs)
    def get_action(self, state, epsilon = 0, return_values = False):
        #print(state.shape)
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        
        else:
            actions = self.predict(state[np.newaxis])
            return np.argmax(actions)

    def format_states_actions(self, state, pred_actions):
        return np.concatenate((pred_actions.reshape((pred_actions.shape[0],-1)), state[:,-1]), axis=-1)

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
            #print(own_weights[k].shape, other_weights[k].shape)
            new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]
        self.set_weights(new_weights)