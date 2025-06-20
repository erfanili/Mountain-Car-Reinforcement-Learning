import numpy as np

from tqdm import trange


def get_epsilon(ep, eps_start=1, eps_end=0.03, decay_episodes=100):
    return max(eps_end, eps_start - (eps_start - eps_end) * (ep / decay_episodes))

def train(env,agent, num_episodes = 500, max_steps = 200):
    episode_returns  = []
    
    for ep in trange(num_episodes,desc="Training"):
        state = env.reset()
        agent.reset()
        agent.epsilon = get_epsilon(ep)
        
        max_pos = state[0]
        action_idx = agent.act(state)
        total_reward = 0.
        
        for t in range(max_steps):
            next_state, reward, done = env.step(action_idx)
            next_action_idx = agent.act(next_state)
            
            max_pos = max(max_pos, next_state[0])
            agent.update(state, action_idx,0.0, next_state, next_action_idx, done)
            state = next_state
            action_idx = next_action_idx
            total_reward += reward
            
            if done:
                break
            
        scaled_reward = max_pos
        reward = max_pos
    

# Bonus if goal is achieved
        if max_pos >= env.goal_position:
            scaled_reward += 1  # or more — make this big and meaningful# shift from -1.2 to 0
        agent.update(state, action_idx, scaled_reward, state, action_idx, True)

        episode_returns.append(scaled_reward)
    
    return episode_returns