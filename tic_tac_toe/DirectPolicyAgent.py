import tensorflow as tf
import numpy as np
from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, NAUGHT, CROSS, GameResult
from tic_tac_toe.Player import Player
from tic_tac_toe.TFSessionManager import TFSessionManager as TFSN
import random


# Tic Tac Toe Policy Gradient Agent
#
# This class implements a Tic Tac Toe playing Neural Network that learns by direct policy gradient descent
# It is largely based on Arthur Juliani's github repo: DeepRL-Agents: https://github.com/awjuliani/DeepRL-Agents
#
# Based on: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb


# The Policy-Based Agent

class PolicyGradientNetwork:
    """
    The Policy Gradient TensorFlow Network
    """

    def __init__(self, name, learning_rate=0.001, beta: float = 0.00001):
        """
        Constructor for the Policy Gradient Network
        :param name: Name and TensorFlow scope of the network.
        :param learning_rate: Learning rate of the TensorFlow Optimizer
        :param beta: Factor multiplied with the Regularization loss
        """
        self.state_in = None
        self.logits = None
        self.output = None
        self.chosen_action = None
        self.reward_holder = None
        self.action_holder = None
        self.indexes = None
        self.responsible_outputs = None
        self.loss = None
        self.update_batch = None
        self.name = name
        self.learning_rate = learning_rate
        self.reg_losses = None
        self.merge = None
        self.beta = beta
        self.build_graph()

    def add_dense_layer(self, input_tensor: tf.Tensor, output_size: int, activation_fn=None,
                        name: str = None) -> tf.Tensor:
        """
        Adds a dense Neural Net layer to network input_tensor
        :param input_tensor: The layer to which we should add the new layer
        :param output_size: The output size of the new layer
        :param activation_fn: The activation function for the new layer, or None if no activation function
        should be used
        :param name: The optional name of the layer. Useful for saving a loading a TensorFlow graph
        :return: A new dense layer attached to the `input_tensor`
        """
        return tf.layers.dense(input_tensor, output_size, activation=activation_fn,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(),
                               name=name)

    def build_graph(self):
        """
        Builds the actual Network Graph
        """

        with tf.variable_scope(self.name):
            self.state_in = tf.placeholder(shape=[None, BOARD_SIZE * 3], dtype=tf.float32)

            hidden = self.add_dense_layer(self.state_in, BOARD_SIZE * 3 * 9, activation_fn=tf.nn.relu)
            # hidden = self.add_dense_layer(hidden, BOARD_SIZE * 3 * 20, activation_fn=tf.nn.relu)
            self.logits = self.add_dense_layer(hidden, 9, activation_fn=None)

            self.output = tf.nn.softmax(self.logits)
            tf.summary.histogram("Action_policy_values", self.output)

            self.chosen_action = tf.argmax(self.output, 1)

            # The next six lines establish the training procedure. We feed the reward and chosen action
            # into the network to compute the loss, and use it to update the network.
            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

            self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
            self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

            self.loss = - tf.reduce_mean(tf.log(self.responsible_outputs + 1e-9) * self.reward_holder)
            tf.summary.scalar("policy_loss", self.loss)

            self.reg_losses = tf.identity(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name),
                                          name="reg_losses")

            reg_loss = self.beta * tf.reduce_mean(self.reg_losses)
            tf.summary.scalar("Regularization_loss", reg_loss)

            self.merge = tf.summary.merge_all(scope=self.name)

            total_loss = tf.add(self.loss, reg_loss, name="total_loss")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.update_batch = optimizer.minimize(total_loss)


class ReplayBuffer:
    """
    This class manages the Experience Replay buffer for the Neural Network player
    """

    def __init__(self, buffer_size=3000):
        """
        Creates a new `ReplayBuffer` of size `buffer_size`.
        :param buffer_size:
        """
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience: []):
        """
        Adds a list of experience Tuples to the buffer. If this operation causes the buffer to be longer than its
        defined maximum, old entries will be evicted until the maximum length is achieved. Entries are added and
        evicated on a FIFO basis.
        :param experience: A list of experience tuples to be added to the replay buffer
        """
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:1] = []
        self.buffer.append(experience)

    def sample(self, size) -> []:
        """
        Returns a random sample of `size` entries from the Replay Buffer. If there are less than `size` entries
        in the buffer, all entries will be returned.
        :param size: Number of sample to be returned
        :return: List of `size` number of randomly sampled  previously stored entries.
        """
        size = min(len(self.buffer), size)
        return random.sample(self.buffer, size)


class DirectPolicyAgent(Player):

    def board_state_to_nn_input(self, state: np.ndarray) -> np.ndarray:
        """
        Converts a Tic Tac Tow board state to an input feature vector for the Neural Network. The input feature vector
        is a bit array of size 27. The first 9 bits are set to 1 on positions containing the player's pieces, the second
        9 bits are set to 1 on positions with our opponents pieces, and the final 9 bits are set on empty positions on
        the board.
        :param state: The board state that is to be converted to a feature vector.
        :return: The feature vector representing the input Tic Tac Toe board state.
        """
        res = np.array([(state == self.side).astype(int),
                        (state == Board.other_side(self.side)).astype(int),
                        (state == EMPTY).astype(int)])
        return res.reshape(-1)

    def __init__(self, name, gamma: float = 0.1, learning_rate: float = 0.001, win_value: float = 1.0,
                 loss_value: float = 0.0, draw_value: float = 0.5, training: bool = True,
                 random_move_probability: float = 0.9, beta: float = 0.000001,
                 random_move_decrease: float = 0.9997, pre_training_games: int = 500, batch_size: int = 60):
        """

        :param name: Name and TensorFlow scope of the agent
        :param gamma: Reward discount factor
        :param learning_rate: Learning rate for the TensorFlow Optimizer
        :param win_value: Reward for winning a game
        :param loss_value: Reward for losing a game
        :param draw_value: Reward for playing a draw
        :param training: Boolean flag to indicate whether the network should train
        :param random_move_probability: Probability of making a random move instead of following policy
        :param beta: Factor multiplied with the Regularization loss
        :param random_move_decrease: Factor by which to decrease the probability of a random move after each game
        :param pre_training_games: How many initial games to be played completely randomly, i.e. policy not applied
        at all
        :param batch_size: Size of training batch sampled from experience buffer for each training run
        """
        super().__init__()
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value

        self.training = training
        self.batch_size = batch_size

        self.random_move_probability = random_move_probability
        self.random_move_decrease = random_move_decrease
        self.pre_training_games = pre_training_games
        self.game_counter = 0
        self.side = None
        self.board_position_log = []
        self.action_log = []

        self.replay_buffer_win = ReplayBuffer()
        self.replay_buffer_loss = ReplayBuffer()
        self.replay_buffer_draw = ReplayBuffer()

        self.name = name
        self.nn = PolicyGradientNetwork(name, learning_rate, beta)
        self.writer = None

    def new_game(self, side: int):
        """
        Reset internal state and prepare to play a new game
        :param side: The side we will play in the coming game
        """
        self.side = side
        self.board_position_log = []
        self.action_log = []

    def get_probs(self, input_pos: [np.ndarray]) -> ([float], [float]):
        """
        Compute action probabilities through the Neural Network
        :param input_pos: List of input states for which to compute probabilities
        :return: Tuple of lists of action probabilities and raw logits for the given input states
        """
        probs, logits = TFSN.get_session().run([self.nn.output, self.nn.logits],
                                               feed_dict={self.nn.state_in: input_pos})
        return probs, logits

    def get_valid_probs(self, input_pos: [np.ndarray], boards: [Board]) -> ([float], [float]):
        """
        Evaluates the board positions `input_pos` with the Neural Network `network`. It post-processes the result
        by setting the probability of all illegal moves in the current position to 0.
        It returns a tuple of post-processed probabilities and q values.
        :param input_pos: The board position to be evaluated as feature vector for the Neural Network
        :param boards: A list of corresponding Board objects for testing if a move is illegal.
        :return: A tuple of post-processed probabilities and q values. Probabilities for illegal moves are set to 0 and
        the sum of all probabilities is 1. Can returns arrays of NaN if all moves have probability 0.
        """
        probabilities, _ = self.get_probs(input_pos)

        probabilities = np.copy(probabilities)

        # We filter out all illegal moves by setting the probability to 0. We don't change the q values
        # as we don't want the NN to waste any effort of learning different Q values for moves that are illegal
        # anyway.
        for prob, b in zip(probabilities, boards):
            for index, p in enumerate(prob):
                if not b.is_legal(index):
                    prob[index] = 0
        res = probabilities / probabilities.sum(axis=1, keepdims=True)
        return res

    def move(self, board: Board) -> (GameResult, bool):
        """
        Makes a move on the given input state
        :param board: The current state of the game
        :return: The GameResult after this move, Flag to indicate whether the move finished the game
        """
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs = self.get_valid_probs([nn_input], [board])
        probs = probs[0]

        # Most of the time our next move is the one with the highest probability after removing all illegal ones.
        # Occasionally, however we randomly chose a random move to encourage exploration
        if (self.training is True) and \
                (self.game_counter < self.pre_training_games):
            move = board.random_empty_spot()
        else:
            if np.isnan(probs).any():  # Can happen when all probabilities degenerate to 0. Best thing we can do is
                # make a random legal move
                move = board.random_empty_spot()
            else:
                move = np.random.choice(np.arange(len(probs)), p=probs)
            if not board.is_legal(move):  # Debug case only, I hope
                print("Illegal move!")

        # We record the action we selected as well as the Q values of the current state for later use when
        # adjusting NN weights.
        self.action_log.append(move)

        _, res, finished = board.move(move, self.side)

        return res, finished

    def add_game_to_replay_buffer(self, final_reward: float, rewards: []):
        """
        Adds the game history of the current game to the replay buffer. This method is called internally
        after the game has finished
        :param rewards: The rewards for all the moves in the game
        :param final_reward: The reward for the final move in the game
        """
        game_length = len(self.action_log)

        if final_reward == self.win_value:
            buffer = self.replay_buffer_win
        elif final_reward == self.loss_value:
            buffer = self.replay_buffer_loss
        else:
            buffer = self.replay_buffer_draw

        for i in range(game_length):
            buffer.add([self.board_position_log[i], self.action_log[i], rewards[i]])

    def calculate_rewards(self, final_reward: float, length: int) -> [float]:
        """
        Computes and returns the discounted rewards for all moves
        :param final_reward: The reward of the final move of the game
        :param length:  The number of moves in the game
        :return: List of discounted rewards for all moves
        """
        discounted_r = np.zeros(length)

        running_add = final_reward
        for t in reversed(range(0, length)):
            discounted_r[t] = running_add
            running_add = running_add * self.gamma
        return discounted_r.tolist()

    def final_result(self, result: GameResult):
        """
        Called when the game has ended. Time to record results and train the network.
        :param result: The final result of the game
        """
        # Compute the final reward based on the game outcome
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or (
                result == GameResult.CROSS_WIN and self.side == CROSS):
            final_reward = self.win_value  # type: float
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or (
                result == GameResult.CROSS_WIN and self.side == NAUGHT):
            final_reward = self.loss_value  # type: float
        elif result == GameResult.DRAW:
            final_reward = self.draw_value  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.game_counter += 1

        rewards = self.calculate_rewards(final_reward, len(self.action_log))

        # noinspection PyTypeChecker
        self.add_game_to_replay_buffer(final_reward, rewards)

        # If we are in training mode we run the optimizer.
        if self.training and (self.game_counter > self.pre_training_games):

            batch_third = self.batch_size // 3
            train_batch = self.replay_buffer_win.sample(batch_third)
            train_batch.extend(self.replay_buffer_loss.sample(batch_third))
            train_batch.extend(self.replay_buffer_draw.sample(batch_third))
            train_batch = np.array(train_batch)

            # We convert the input states we have recorded to feature vectors to feed into the training.
            nn_input = np.array([self.board_state_to_nn_input(x[0]) for x in train_batch])
            actions = np.array(train_batch[:, 1])
            rewards = np.array(train_batch[:, 2])
            feed_dict = {self.nn.reward_holder: rewards,
                         self.nn.action_holder: actions,
                         self.nn.state_in: nn_input}
            summary, _, inds, rps, loss = TFSN.get_session().run([self.nn.merge, self.nn.update_batch,
                                                                  self.nn.indexes, self.nn.responsible_outputs,
                                                                  self.nn.loss], feed_dict=feed_dict)

            self.random_move_probability *= self.random_move_decrease

            if self.writer is not None:
                self.writer.add_summary(summary, self.game_counter)
                summary = tf.Summary(value=[tf.Summary.Value(tag='Random_Move_Probability',
                                                             simple_value=self.random_move_probability)])
                self.writer.add_summary(summary, self.game_counter)
