import numpy as np
from collections import defaultdict


class TileAgent:
    def __init__(self,n_actions, tile_coder, alpha =0.1, gamma = 1., lam = 0.9, epsilon = 1):
        self.n_actions = n_actions
        self.tile_coder = tile_coder
        self.alpha = alpha/self.tile_coder.num_tilings
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        
        self.n_features = self.tile_coder.feature_vector_length()
        self.weights = np.zeros((n_actions, self.n_features))
        self.z = np.zeros((n_actions,self.n_features))
        
        
    def get_q(self,state):
        position, velocity = state
        features = self.tile_coder.get_features(position, velocity)
        return np.array([np.sum([self.weights[a][features]]) for a in range(self.n_actions)])
    
    
    def act(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        q_values = self.get_q(state)
        
        return np.argmax(q_values)
        
        
    def update(self, state, action_idx, reward, next_state, next_action_idx,done):
        features = self.tile_coder.get_features(*state)
        next_features = self.tile_coder.get_features(*next_state)
        
        
        q_sa = np.sum(self.weights[action_idx][features])
        q_s_next_a = 0 if done else np.sum(self.weights[next_action_idx][next_features])
        
        delta = reward + self.gamma* q_s_next_a - q_sa
        
        
        self.z *= self.gamma * self.lam
        self.z[action_idx][features] += 1
        
        self.weights += self.alpha * delta * self.z
        
        if done:
            self.z[:] = 0
            
    def reset(self):
        self.z[:] = 0
        
    def save(self, path):
        np.save(path, self.weights)

    def load(self, path):
        self.weights = np.load(path)