import edugrid  # registers the environment
from edugrid.envs import Action
import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    env = gym.make(
        "philsteg/EduGrid-v0",
        size=(3, 3),
        agent_location=(0, 0),
        wall_locations=[(2, slice(None))],
        sink_locations=[(0, 1), (1, 0)],
        target_locations=[(0, 2)],
        slip_prob=0.5,
    )
    # Inspect the transition matrix shape
    print(
        f"transition matrix shape: {env.unwrapped.transition_matrix.shape}"
    )  # (rows, columns, actions, rows, columns) = (3, 3, 4, 3, 3)

    # Modify the transition matrix so that the agent always transitions from (0, 1) to (0, 2) if the action "down" is chosen.
    new_transition_prob = np.zeros((3, 3))
    new_transition_prob[0, 2] = 1
    env.unwrapped.transition_matrix[0, 1, Action.DOWN] = new_transition_prob

    # Inspect the reward matrix shape
    print(
        f"reward matrix shape: {env.unwrapped.reward_matrix.shape}"
    )  # (rows, columns, actions, rows, columns) = (3, 3, 4, 3, 3)

    # Modify the reward matrix so that the agent receives a penalty of -10 if the sink at (1, 0) is reached.
    env.unwrapped.reward_matrix[:, :, :, 1, 0] = -10

    # Inspect the terminal matrix shape
    print(
        f"terminal matrix shape: {env.unwrapped.terminal_matrix.shape}"
    )  # (rows, columns) = (3, 3)

    # Modify the terminal matrix so that (1, 2) is terminal.
    env.unwrapped.terminal_matrix[1, 2] = True
