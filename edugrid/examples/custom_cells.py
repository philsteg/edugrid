import edugrid  # registers the environment
from edugrid.envs.cells import Cell
import gymnasium as gym


class CustomCell(Cell):

    def __init__(self):
        self.is_terminal = False

    def is_blocking(self):
        return False

    def on_entered(self, env, row, column):
        print(f"Custom cell at ({row}, {column}) entered.")

    def on_left(self, env, row, column):
        print(f"Custom cell at ({row}, {column}) left.")

    def on_step(self, env, row, column):
        # Change each step whether this cell is terminal
        self.is_terminal = not self.is_terminal
        env.terminal_matrix[row, column] = self.is_terminal
        env.reward_matrix[:, :, :, row, column] = 5 if self.is_terminal else -1
        print(
            f"On-step update for custom cell at ({row}, {column}): Cell is {'' if self.is_terminal else 'NOT '}terminal."
        )


if __name__ == "__main__":
    custom_cells = {
        (0, 4): CustomCell(),
        (4, 0): CustomCell(),
    }
    env = gym.make("philsteg/EduGrid-v0", custom_cells=custom_cells)

    obs, info = env.reset()

    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"Terminated at cell ({obs['agent'][0]}, {obs['agent'][1]})")

        if terminated or truncated:
            break
