import datetime
import pathlib

import gym
import numpy
import torch

from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (3, 6, 6)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(36))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 36  # Maximum number of moves if game is not finished before
        self.num_simulations = 25  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 36  # Number of game moves to keep for every batch element
        self.td_steps = 36  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Hex()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 6 + (col - 1)
                if (
                    choice in self.legal_actions()
                    and 1 <= row
                    and 1 <= col
                    and row <= 6
                    and col <= 6
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 6 + 1
        col = action_number % 6 + 1
        return f"Play row {row}, column {col}"


class Hex:
    def __init__(self):
        self.size = 6
        self.board = numpy.zeros((6, 6), dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((6, 6), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        row = action // 6
        col = action % 6
        self.board[row, col] = self.player

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), 10 * reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        board_to_play = numpy.full((6, 6), self.player)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        legal = []
        for i in range(36):
            row = i // 6
            col = i % 6
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def get_neighbours(self, x, y):
        # Get the neighbouring coordinates of the current cell
        neighbours = []
        if x > 0:
            neighbours.append((x - 1, y))
        if x < self.size - 1:
            neighbours.append((x + 1, y))
        if y > 0:
            neighbours.append((x, y - 1))
        if y < self.size - 1:
            neighbours.append((x, y + 1))
        if x > 0 and y < self.size - 1:
            neighbours.append((x - 1, y + 1))
        if x < self.size - 1 and y > 0:
            neighbours.append((x + 1, y - 1))
        return neighbours

    def dfs(self, i, j, visited):
        if i < 0 or i >= self.size or j < 0 or j >= self.size:
            return False
        if self.board[i, j] != self.player:
            return False
        if visited[i, j]:
            return False
        if j == self.size - 1 and self.player == 1:
            return True
        if i == self.size - 1 and self.player == -1:
            return True
        visited[i, j] = True
        for ni, nj in self.get_neighbours(i, j):
            if self.dfs(ni, nj, visited):
                return True
        return False

    def have_winner(self):
        visited = numpy.zeros((self.size, self.size), dtype=bool)

        if self.player == 1:
            for i in range(self.size):
                if self.dfs(i, 0, visited):
                    return True
            return False
        else:
            for j in range(self.size):
                if self.dfs(0, j, visited):
                    return True
            return False

    def render(self):
        print("\033[34m _ _ _ _ _ _\033[0m")
        markers = ["\033[32m|\033[0m", "\033[32m |\033[0m", "\033[32m  |\033[0m", "\033[32m   |\033[0m", "\033[32m    |\033[0m", "\033[32m     |\033[0m"]

        for row in range(6):
            line = list("\033[32mA\033[0m" if x == 1 else "\033[34mB\033[0m" if x == -1 else "*" for x in self.board[row])
            print(markers[row] + ' '.join(map(str, line)))

        print()
