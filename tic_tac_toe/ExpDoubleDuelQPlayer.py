#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
# Based on https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
#

import numpy as np
import random
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

        # Placeholders

        self.input_positions = None
        self.target_q = None
        self.actions = None

        # Internal tensors
        self.actions_onehot = None
        self.value = None
        self.advantage = None

        self.td_error = None
        self.q = None
        self.loss = None

        # Externally useful tensors

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
            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32, name='target')

            net = self.input_positions

            net = self.add_dense_layer(net, BOARD_SIZE * 3 * 9, tf.nn.relu)

            self.value = self.add_dense_layer(net, 1, name='value')
            self.advantage = self.add_dense_layer(net, BOARD_SIZE, name='advantage')

            self.q_values = self.value + tf.subtract(self.advantage,
                                                     tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            self.probabilities = tf.nn.softmax(self.q_values, name='probabilities')

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
            self.actions_onehot = tf.one_hot(self.actions, BOARD_SIZE, dtype=tf.float32)
            self.q = tf.reduce_sum(tf.multiply(self.q_values, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.target_q - self.q)
            self.loss = tf.reduce_mean(self.td_error)

            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(self.loss,
                                                                                                          name='train')


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


class ExpDoubleDuelQPlayer(Player):
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

    def create_graph_copy_op(self, src: str, target: str, tau: float) -> [tf.Tensor]:
        """
        Creates and returns a TensorFlow Operation that copies the content of all trainable variables from the
        sub-graph in scope `src` to the sub-graph in scope `target`. Both graphs need to have the same topology and
        the trainable variable been added in the same order for this to work.

        The value `tau` determines to which degree the src value will replace the target value according to the
        foumla: nee_value = src * (1-tau) + target * tau
        :param src: The name of the scope from which to copy the variables
        :param target: The name of the scope to which the variables are copied
        :param tau: A float value between 0 and 1 which determines the weight of src and target for the new value
        :return: A list of TensorFlow tensors for the copying operations
        """
        src_vars = tf.trainable_variables(src)
        target_vars = tf.trainable_variables(target)

        op_holder = []

        for s, t in zip(src_vars, target_vars):
            op_holder.append(t.assign((s.value() * tau) + ((1 - tau) * t.value())))
        return op_holder

    def __init__(self, name: str, reward_discount: float = 0.95, win_value: float = 1.0, draw_value: float = 0.0,
                 loss_value: float = -1.0, learning_rate: float = 0.01, training: bool = True,
                 random_move_prob: float = 0.95, random_move_decrease: float = 0.95, batch_size=50,
                 pre_training_games: int = 500, tau: float = 0.001):
        """
        Constructor for the Neural Network player.
        :param batch_size: The number of samples from the Experience Replay Buffer to be used per training operation
        :param pre_training_games: The number of games played completely radnomly before using the Neural Network
        :param tau: The factor by which the target Q graph gets updated after each training operation
        :param name: The name of the player. Also the name of its TensorFlow scope. Needs to be unique
        :param reward_discount: The factor by which we discount the maximum Q value of the following state
        :param win_value: The reward for winning a game
        :param draw_value: The reward for playing a draw
        :param loss_value: The reward for losing a game
        :param learning_rate: The learning rate of the Neural Network
        :param training: Flag indicating if the Neural Network should adjust its weights based on the game outcome
        (True), or just play the game without further adjusting its weights (False).
        :param random_move_prob: Initial probability of making a random move
        :param random_move_decrease: Factor by which to decrease of probability of random moves after a game
        """
        self.tau = tau
        self.batch_size = batch_size
        self.reward_discount = reward_discount
        self.win_value = win_value
        self.draw_value = draw_value
        self.loss_value = loss_value
        self.side = None
        self.board_position_log = []
        self.action_log = []
        self.next_state_log = []

        self.name = name
        self.q_net = QNetwork(name + '_main', learning_rate)
        self.target_net = QNetwork(name + '_target', learning_rate)

        self.graph_copy_op = self.create_graph_copy_op(name + '_main', name + '_target', self.tau)
        self.training = training
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease

        self.replay_buffer_win = ReplayBuffer()
        self.replay_buffer_loss = ReplayBuffer()
        self.replay_buffer_draw = ReplayBuffer()

        self.game_counter = 0
        self.pre_training_games = pre_training_games

        super().__init__()

    def new_game(self, side: int):
        """
        Prepares for a new games. Store which side we play and clear internal data structures for the last game.
        :param side: The side it will play in the new game.
        """
        self.side = side
        self.board_position_log = []
        self.action_log = []

    def add_game_to_replay_buffer(self, reward: float):
        """
        Adds the game history of the current game to the replay buffer. This method is called internally
        after the game has finished
        :param reward: The reward for the final move in the game
        """
        game_length = len(self.action_log)

        if reward == self.win_value:
            buffer = self.replay_buffer_win
        elif reward == self.loss_value:
            buffer = self.replay_buffer_loss
        else:
            buffer = self.replay_buffer_draw

        for i in range(game_length - 1):
            buffer.add([self.board_position_log[i], self.action_log[i],
                        self.board_position_log[i + 1], 0])

        buffer.add([self.board_position_log[game_length - 1], self.action_log[game_length - 1], None, reward])

    def get_probs(self, input_pos: [np.ndarray], network: QNetwork) -> ([float], [float]):
        """
        Feeds the feature vectors `input_pos` (which encode a board states) into the Neural Network and computes the
        Q values and corresponding probabilities for all moves (including illegal ones).
        :param network: The network to get probabilities from
        :param input_pos: A list of feature vectors to be fed into the Neural Network.
        :return: A list of tuples of probabilities and q values of all actions (including illegal ones).
        """
        probs, qvalues = TFSN.get_session().run([network.probabilities, network.q_values],
                                                feed_dict={network.input_positions: input_pos})
        return probs, qvalues

    def get_valid_probs(self, input_pos: [np.ndarray], network: QNetwork, boards: [Board]) -> ([float], [float]):
        """
        Evaluates the board positions `input_pos` with the Neural Network `network`. It post-processes the result
        by setting the probability of all illegal moves in the current position to -1.
        It returns a tuple of post-processed probabilities and q values.
        :param input_pos: The board position to be evaluated as feature vector for the Neural Network
        :param network: The Neural Network
        :param boards: A list of corresponding Board objects for testing if a move is illegal.
        :return: A tuple of post-processed probabilities and q values. Probabilities for illegal moves are set to -1.
        """
        probabilities, qvals = self.get_probs(input_pos, network)
        qvals = np.copy(qvals)
        probabilities = np.copy(probabilities)

        # We filter out all illegal moves by setting the probability to 0. We don't change the q values
        # as we don't want the NN to waste any effort of learning different Q values for moves that are illegal
        # anyway.
        for q, prob, b in zip(qvals, probabilities, boards):
            for index, p in enumerate(q):
                if not b.is_legal(index):
                    prob[index] = -1
                elif prob[index] < 0:
                    prob[index] = 0.0

        return probabilities, qvals

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
        probs, _ = self.get_valid_probs([nn_input], self.q_net, [board])
        probs = probs[0]

        # Most of the time our next move is the one with the highest probability after removing all illegal ones.
        # Occasionally, however we randomly chose a random move to encourage exploration

        # noinspection PyUnresolvedReferences
        if (self.training is True) and \
                ((self.game_counter < self.pre_training_games) or (np.random.rand(1) < self.random_move_prob)):
            move = board.random_empty_spot()
        else:
            move = np.argmax(probs)

        # We record the action we selected as well as the Q values of the current state for later use when
        # adjusting NN weights.
        self.action_log.append(move)

        # We execute the move and return the result
        _, res, finished = board.move(move, self.side)
        return res, finished

    def final_result(self, result: GameResult):
        """
        This method is called once the game is over. If `self.training` is True, we execute a training run for
        the Neural Network.
        :param result: The result of the game that just finished.
        """

        self.game_counter += 1

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

        self.add_game_to_replay_buffer(reward)

        # If we are in training mode we run the optimizer.
        if self.training and (self.game_counter > self.pre_training_games):

            batch_third = self.batch_size // 3
            train_batch = self.replay_buffer_win.sample(batch_third)
            train_batch.extend(self.replay_buffer_loss.sample(batch_third))
            train_batch.extend(self.replay_buffer_draw.sample(batch_third))
            train_batch = np.array(train_batch)

            #
            # Let's compute the target q values for all non terminal move
            # We extract the resulting state, run it through the target net work and
            # get the maximum q value (of all valid moves)
            next_states = [s[2] for s in train_batch if s[2] is not None]
            target_qs = []

            if len(next_states) > 0:
                probs, qvals = self.get_valid_probs([self.board_state_to_nn_input(s) for s in next_states],
                                                    self.target_net, [Board(s) for s in next_states])

                i = 0
                for t in train_batch:
                    if t[2] is not None:
                        max_move = np.argmax(probs[i])
                        max_qval = qvals[i][max_move]
                        target_qs.append(max_qval * self.reward_discount)
                        i += 1
                    else:
                        target_qs.append(t[3])

                if i != len(next_states):
                    print("Something wrong here!!!")
            else:
                target_qs.extend(train_batch[:, 3])

            # We convert the input states we have recorded to feature vectors to feed into the training.
            nn_input = [self.board_state_to_nn_input(x[0]) for x in train_batch]
            actions = train_batch[:, 1]
            # We run the training step with the recorded inputs and new Q value targets.
            TFSN.get_session().run([self.q_net.train_step],
                                   feed_dict={self.q_net.input_positions: nn_input,
                                              self.q_net.target_q: target_qs,
                                              self.q_net.actions: actions})

            TFSN.get_session().run(self.graph_copy_op)

            self.random_move_prob *= self.random_move_decrease
