from env.mountain_car import MountainCarEnv
from utils.tile_coder import TileCoder
from agent.tile_agent import TileAgent
from train import train
import matplotlib.pyplot as plt
import numpy as np



def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')


def main():
    # 1. Create environment
    env = MountainCarEnv()

    # 2. Create tile coder
    tile_coder = TileCoder(
        position_range=env.position_range,
        velocity_range=env.velocity_range,
        num_tilings=8,
        tiles_per_dim=8,
    )

    # 3. Create agent
    agent = TileAgent(
        n_actions=3,
        tile_coder=tile_coder,
        alpha=0.1,
        gamma=1,
        lam=0.99,
        epsilon=1,
    )

    # 4. Train
    num_episodes = 5000
    returns = train(env, agent, num_episodes=num_episodes)
    
    
    agent.save("data/tile_agent_weights.npy")
    

    # 5. Plot results
    plt.plot(smooth(returns))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("SARSA(λ) with Tile Coding on MountainCar")
    plt.grid(True)
    plt.show()
    plt.savefig(f'data/train_rewards.png')

if __name__ == "__main__":
    main()
