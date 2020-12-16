import numpy as np

class Average_ensemble():
    def __init__(self, actions, models, normalize_preds = False, *args, **kwargs):
        self.actions = actions
        self.models = models
        self.n_models = len(self.models)
        self.n_actions = len(self.actions)
        self.normalize_preds = normalize_preds
       
    def get_action(self, state, return_values = False):
        
        pred_actions = np.zeros((self.n_models, self.n_actions))
        for i in range(self.n_models):
            a = self.models[i].predict(state[np.newaxis, -self.models[i].network.memory_size:])

            if self.normalize_preds:
                a = a / np.linalg.norm(a, axis=-1)

            pred_actions[i] = a
            
            
        avg_actions = np.mean(pred_actions, axis=0)
        if return_values:
            return np.argmax(avg_actions), avg_actions, pred_actions
        return np.argmax(avg_actions)