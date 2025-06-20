import numpy as np

class MountainCarEnv:
    def __init__(self):
        self.position_range = (-1.2,0.6)
        self.velocity_range = (-0.7,0.7)
        self.goal_position = 0.5
        
        self.acc = [-.01, 0., .01]
        
        self.reset()
        
    def reset(self):
        self.position = np.random.uniform(-0.6,-0.4)
        self.velocity = 0.
        
        return np.array([self.position, self.velocity],dtype=np.float32)
    
    
    def step(self,action_idx):
        acc = self.acc[action_idx]
        self.velocity += acc - 0.025 * np.cos(3*self.position)
        self.velocity = np.clip(self.velocity, *self.velocity_range)
        self.position += self.velocity
        self.position = np.clip(self.position,*self.position_range)
        
        done = self.position > self.goal_position
        
        reward = 0. if done else -1.
        
        return np.array([self.position, self.velocity],dtype=np.float32), reward, done
    
    def get_state(self):
        return np.array([self.position, self.velocity],dtype=np.float32)
            
        