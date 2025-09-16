import numpy as np
import pytest

from edugrid.algorithms.policy import DeterministicPolicy, StochasticPolicy
from edugrid.envs.grids import Action, EduGridEnv

RIGHT = np.array([1, 0, 0, 0])
UP = np.array([0, 1, 0, 0])
LEFT = np.array([0, 0, 1, 0])
DOWN = np.array([0, 0, 0, 1])
ALL = np.array([0.25, 0.25, 0.25, 0.25])
UP_LEFT = np.array([0, 0.5, 0.5, 0])
UP_RIGHT = np.array([0.5, 0.5, 0, 0])
DOWN_LEFT = np.array([0, 0, 0.5, 0.5])
DOWN_RIGHT = np.array([0.5, 0, 0, 0.5])


@pytest.fixture
def simple_env():
    return EduGridEnv(config="config_5x5_v0")


@pytest.fixture
def simple_transitions() -> np.ndarray:
    # shape: (ncols, nrows, num_actions, ncols, nrows)

    m = np.zeros((5, 5, 5, 5))

    for i in range(5):
        for j in range(5):
            a = np.zeros((5, 5))
            a[i, j] = 1.0
            m[i, j] = a

    # Actions: right, up, left, down
    transitions = np.array(
        [
            [  # (0,...)
                [m[0, 1], m[0, 0], m[0, 0], m[1, 0]],
                [m[0, 2], m[0, 1], m[0, 0], m[1, 1]],
                [m[0, 3], m[0, 2], m[0, 1], m[1, 2]],
                [m[0, 4], m[0, 3], m[0, 2], m[1, 3]],
                [m[0, 4], m[0, 4], m[0, 3], m[1, 4]],
            ],
            [  # (1,...)
                [m[1, 1], m[0, 0], m[1, 0], m[2, 0]],
                [m[1, 2], m[0, 1], m[1, 0], m[2, 1]],
                [m[1, 3], m[0, 2], m[1, 1], m[2, 2]],
                [m[1, 4], m[0, 3], m[1, 2], m[2, 3]],
                [m[1, 4], m[0, 4], m[1, 3], m[2, 4]],
            ],
            [  # (2,...)
                [m[2, 1], m[1, 0], m[2, 0], m[3, 0]],
                [m[2, 2], m[1, 1], m[2, 0], m[3, 1]],
                [m[2, 3], m[1, 2], m[2, 1], m[3, 2]],
                [m[2, 4], m[1, 3], m[2, 2], m[3, 3]],
                [m[2, 4], m[1, 4], m[2, 3], m[3, 4]],
            ],
            [  # (3,...)
                [m[3, 1], m[2, 0], m[3, 0], m[4, 0]],
                [m[3, 2], m[2, 1], m[3, 0], m[4, 1]],
                [m[3, 3], m[2, 2], m[3, 1], m[4, 2]],
                [m[3, 4], m[2, 3], m[3, 2], m[4, 3]],
                [m[3, 4], m[2, 4], m[3, 3], m[4, 4]],
            ],
            [  # (4,...)
                [m[4, 1], m[3, 0], m[4, 0], m[4, 0]],
                [m[4, 2], m[3, 1], m[4, 0], m[4, 1]],
                [m[4, 3], m[3, 2], m[4, 1], m[4, 2]],
                [m[4, 4], m[3, 3], m[4, 2], m[4, 3]],
                [m[4, 4], m[3, 4], m[4, 3], m[4, 4]],
            ],
        ]
    )
    # adjust for terminal states: (4,4)
    a = np.zeros((5, 5))
    a[4, 4] = 1
    transitions[4, 4, :] = a

    return transitions


@pytest.fixture
def simple_rewards() -> np.ndarray:
    # shape: (ncols, nrows, num_actions, ncols, nrows)
    # normal_reward = -1
    # sink_reward = -5
    # target_reward = 10
    # (4,4) is target

    rewards = -np.ones((5, 5, 4, 5, 5), dtype=np.float64)
    rewards[:, :, :, 4, 4] = 10.0 * np.ones((5, 5, 4))
    rewards[4, 4] = np.zeros((4, 5, 5))
    return rewards


@pytest.fixture
def det_policy():
    policy = DeterministicPolicy((5, 5), 4)
    policy.map = np.array(
        [
            [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.RIGHT],
            [Action.DOWN, Action.LEFT, Action.UP, Action.RIGHT, Action.UP],
            [Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.LEFT],
            [Action.DOWN, Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN],
            [Action.UP, Action.LEFT, Action.DOWN, Action.UP, Action.UP],
        ]
    )
    return policy


@pytest.fixture
def lecture_det_policy():
    policy = DeterministicPolicy((4, 4), 4)
    policy.map = np.array(
        [
            [Action.RIGHT, Action.LEFT, Action.LEFT, Action.LEFT],
            [Action.UP, Action.UP, Action.LEFT, Action.DOWN],
            [Action.UP, Action.RIGHT, Action.RIGHT, Action.DOWN],
            [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT],
        ]
    )
    return policy


@pytest.fixture
def lecture_uniform_stoch_policy():
    policy = StochasticPolicy((4, 4), 4)
    return policy


@pytest.fixture
def lecture_stoch_policy():
    policy = StochasticPolicy((4, 4), 4)
    policy.probs = np.array(
        [
            [ALL, LEFT, LEFT, DOWN_LEFT],
            [UP, UP_LEFT, DOWN_LEFT, DOWN],
            [UP, UP_RIGHT, DOWN_RIGHT, DOWN],
            [UP_RIGHT, RIGHT, RIGHT, ALL],
        ]
    )
    return policy


@pytest.fixture
def lecture_opt_det_policy():
    policy = DeterministicPolicy((4, 4), 4)
    policy.map = np.array(
        [
            [Action.RIGHT, Action.LEFT, Action.LEFT, Action.LEFT],
            [Action.UP, Action.UP, Action.RIGHT, Action.DOWN],
            [Action.UP, Action.RIGHT, Action.RIGHT, Action.DOWN],
            [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT],
        ]
    )
    return policy


@pytest.fixture
def lecture_opt_stoch_policy():
    policy = StochasticPolicy((4, 4), 4)

    policy.probs = np.array(
        [
            [ALL, LEFT, LEFT, DOWN_LEFT],
            [UP, UP_LEFT, ALL, DOWN],
            [UP, ALL, DOWN_RIGHT, DOWN],
            [UP_RIGHT, RIGHT, RIGHT, ALL],
        ]
    )
    return policy


@pytest.fixture
def lecture_action_values():
    values = np.array(
        [
            [
                [0, 0, 0, 0],
                [-21, -15, -1, -19],
                [-23, -21, -15, -21],
                [-23, -23, -21, -21],
            ],
            [
                [-19, -1, -15, -21],
                [-21, -15, -15, -21],
                [-21, -21, -19, -19],
                [-21, -23, -21, -15],
            ],
            [
                [-21, -15, -21, -23],
                [-19, -19, -21, -21],
                [-15, -21, -21, -15],
                [-15, -21, -19, -1],
            ],
            [
                [-21, -21, -23, -23],
                [-15, -21, -23, -21],
                [-1, -19, -21, -15],
                [0, 0, 0, 0],
            ],
        ]
    )
    return values


@pytest.fixture
def lecture_opt_state_values():
    values = np.array(
        [
            [-0, -1, -2, -3],
            [-1, -2, -3, -2],
            [-2, -3, -2, -1],
            [-3, -2, -1, -0],
        ]
    )
    return values


@pytest.fixture
def lecture_opt_action_values():
    values = np.array(
        [
            [
                [-0, -0, -0, -0],
                [-3, -2, -1, -3],
                [-4, -3, -2, -4],
                [-4, -4, -3, -3],
            ],
            [
                [-3, -1, -2, -3],
                [-4, -2, -2, -4],
                [-3, -3, -3, -3],
                [-3, -4, -4, -2],
            ],
            [
                [-4, -2, -3, -4],
                [-3, -3, -3, -3],
                [-2, -4, -4, -2],
                [-2, -3, -3, -1],
            ],
            [
                [-3, -3, -4, -4],
                [-2, -4, -4, -3],
                [-1, -3, -3, -2],
                [-0, -0, -0, -0],
            ],
        ]
    )
    return values


@pytest.fixture
def stoch_policy():
    policy = StochasticPolicy((5, 5), 4, init_mode="uniform")
    return policy


@pytest.fixture
def lecture_env():
    env = EduGridEnv(
        size=(4, 4),
        agent_location=(1, 1),
        target_locations=[(0, 0), (3, 3)],
        target_reward=-1,
    )
    return env
