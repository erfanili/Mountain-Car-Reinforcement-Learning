import matplotlib.pyplot as plt
import numpy as np
from env.mountain_car import MountainCarEnv
from utils.tile_coder import TileCoder
from agent.tile_agent import TileAgent

def evaluate():
    env = MountainCarEnv()
    tile_coder = TileCoder(env.position_range, env.velocity_range, num_tilings=8, tiles_per_dim=8)
    agent = TileAgent(n_actions=3, tile_coder=tile_coder, epsilon=0.0)
    agent.load("data/tile_agent_weights.npy")

    state = env.reset()
    positions = [state[0]]
    forces = []
    timesteps = [0]

    done = False
    t = 0
    while not done:
        action = agent.act(state)
        force = env.acc[action]  # Get actual force value
        
        state, _, done = env.step(action)

        t += 1
        timesteps.append(t)
        positions.append(state[0])
        forces.append(force)

    # --- Plot ---
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Position", color="tab:blue")
    ax1.plot(timesteps, positions, color="tab:blue", label="Position")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = ax1.twinx()  # second axis sharing the same x
    ax2.set_ylabel("Force", color="tab:red")
    ax2.step(timesteps[:-1], forces, where='post', color="tab:red", label="Force")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("MountainCar: Position and Force Over Time")
    fig.tight_layout()
    plt.show()
    plt.savefig('data/eval.png')

if __name__ == "__main__":
    evaluate()
