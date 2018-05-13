#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
# Very trivial NN, but already learns and wins more than it loses against Random Player
#

import numpy as np
import tensorflow as tf
from tic_tac_toe.TFSessionManager import TFSessionManager as TFSN

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult


class QNetwork:
    """
    Contains a TensorFlow graph which is suitable for learning the Tic Tac Toe Q function
    """

    def __init__(self, name: str, learning_rate: float):
        """
        Constructor for QNetwork. Takes a name and a learning rate for the GradientDescentOptimizer
        :param name: Name of the network
        :param learning_rate: Learning rate for the GradientDescentOptimizer
        """
        self.learningRate = learning_rate
        self.name = name
        self.input_positions = None
        self.target_input = None
        self.q_values = None
        self.probabilities = None
        self.train_step = None
        self.build_graph(name)

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
                               name=name)

    def build_graph(self, name: str):
        """
        Builds a new TensorFlow graph with scope `name`
        :param name: The scope for the graph. Needs to be unique for the session.
        """
        with tf.variable_scope(name):
            self.input_positions = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE * 3), name='inputs')

            self.target_input = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE), name='targets')

            net = self.input_positions

            net = self.add_dense_layer(net, BOARD_SIZE * 3 * 9, tf.nn.relu)

            self.q_values = self.add_dense_layer(net, BOARD_SIZE, name='q_values')

            self.probabilities = tf.nn.softmax(self.q_values, name='probabilities')
            mse = tf.losses.mean_squared_error(predictions=self.q_values, labels=self.target_input)
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(mse,
                                                                                                          name='train')


class NNQPlayer(Player):
    """
    Implements a Tic Tac Toe player based on a Reinforcement Neural Network learning the Tic Tac Toe Q function
    """

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

    def __init__(self, name: str, reward_discount: float = 0.95, win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.01, training: bool = True):
        """
        Constructor for the Neural Network player.
        :param name: The name of the player. Also the name of its TensorFlow scope. Needs to be unique
        :param reward_discount: The factor by which we discount the maximum Q value of the following state
        :param win_value: The reward for winning a game
        :param draw_value: The reward for playing a draw
        :param loss_value: The reward for losing a game
        :param learning_rate: The learning rate of the Neural Network
        :param training: Flag indicating if the Neural Network should adjust its weights based on the game outcome
        (True), or just play the game without further adjusting its weights (False).
        """
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.side = None
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.values_log = []
        self.name = name
        self.nn = QNetwork(name, learning_rate)
        self.training = training
        super().__init__()

    def new_game(self, side: int):
        """
        Prepares for a new games. Store which side we play and clear internal data structures for the last game.
        :param side: The side it will play in the new game.
        """
        self.side = side
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.values_log = []

    def calculate_targets(self) -> [np.ndarray]:
        """
        Based on the recorded moves, compute updated estimates of the Q values for the network to learn
        """
        game_length = len(self.action_log)
        targets = []

        for i in range(game_length):
            target = np.copy(self.values_log[i])

            target[self.action_log[i]] = self.reward_discount * self.next_max_log[i]
            targets.append(target)

        return targets

    def get_probs(self, input_pos: np.ndarray) -> ([float], [float]):
        """
        Feeds the feature vector `input_pos` which encodes a board state into the Neural Network and computes the
        Q values and corresponding probabilities for all moves (including illegal ones).
        :param input_pos: The feature vector to be fed into the Neural Network.
        :return: A tuple of probabilities and q values of all actions (including illegal ones).
        """
        probs, qvalues = TFSN.get_session().run([self.nn.probabilities, self.nn.q_values],
                                                feed_dict={self.nn.input_positions: [input_pos]})
        return probs[0], qvalues[0]

    def move(self, board: Board) -> (GameResult, bool):
        """
        Implements the Player interface and makes a move on Board `board`
        :param board: The Board to make a move on
        :return: A tuple of the GameResult and a flag indicating if the game is over after this move.
        """

        # We record all game positions to feed them into the NN for training with the corresponding updated Q
        # values.
        self.board_position_log.append(board.state.copy())

        nn_input = self.board_state_to_nn_input(board.state)
        probs, qvalues = self.get_probs(nn_input)
        qvalues = np.copy(qvalues)

        # We filter out all illegal moves by setting the probability to -1. We don't change the q values
        # as we don't want the NN to waste any effort of learning different Q values for moves that are illegal
        # anyway.
        for index, p in enumerate(qvalues):
            if not board.is_legal(index):
                probs[index] = -1

        # Our next move is the one with the highest probability after removing all illegal ones.
        move = np.argmax(probs)  # int

        # Unless this is the very first move, the Q values of the selected move is also the max Q value of
        # the move that got the game from the previous state to this one.
        if len(self.action_log) > 0:
            self.next_max_log.append(qvalues[move])

        # We record the action we selected as well as the Q values of the current state for later use when
        # adjusting NN weights.
        self.action_log.append(move)
        self.values_log.append(qvalues)

        # We execute the move and return the result
        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        """
        This method is called once the game is over. If `self.training` is True, we execute a training run for
        the Neural Network.
        :param result: The result of the game that just finished.
        """

        # Compute the final reward based on the game outcome
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or (
                result == GameResult.CROSS_WIN and self.side == CROSS):
            reward = self.win_value  # type: float
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or (
                result == GameResult.CROSS_WIN and self.side == NAUGHT):
            reward = self.loss_value  # type: float
        elif result == GameResult.DRAW:
            reward = self.draw_value  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        # The final reward is also the Q value we want to learn for the action that led to it.
        self.next_max_log.append(reward)

        # If we are in training mode we run the optimizer.
        if self.training:
            # We calculate our new estimate of what the true Q values are and feed that into the network as
            # learning target
            targets = self.calculate_targets()

            # We convert the input states we have recorded to feature vectors to feed into the training.
            nn_input = [self.board_state_to_nn_input(x) for x in self.board_position_log]

            # We run the training step with the recorded inputs and new Q value targets.
            TFSN.get_session().run([self.nn.train_step],
                                   feed_dict={self.nn.input_positions: nn_input, self.nn.target_input: targets})
