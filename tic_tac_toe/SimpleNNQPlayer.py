#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
# Very trivial NN, but already learns ane wins more than it loses against Random Player
#

import numpy as np
import tensorflow as tf
from tic_tac_toe.TFSessionManager import TFSessionManager as tfsn

from tic_tac_toe.Board import Board, BOARD_SIZE, EMPTY, CROSS, NAUGHT
from tic_tac_toe.Player import Player, GameResult


class QNetwork:

    def __init__(self, name, learning_rate):
        self.learningRate = learning_rate
        self.name = name
        self.input_positions = None
        self.target_input = None
        self.q_values = None
        self.probabilities = None
        self.train_step = None
        self.build_graph(name)

    def add_dense_layer(self, input_tensor, output_size, activation_fn=None, name=None):
        input_tensor_size = input_tensor.shape[1].value

        weights = tf.Variable(tf.zeros([input_tensor_size, output_size], tf.float32), name='weights')
        bias = tf.Variable(tf.zeros([1, output_size], tf.float32), name='bias')
        layer = tf.matmul(input_tensor, weights) + bias

        if activation_fn is not None:
            layer = activation_fn(layer)

        if name is not None:
            layer = tf.identity(layer, name)

        return layer

    def build_graph(self, name):
        with tf.variable_scope(name):
            self.input_positions = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE * 3), name='inputs')

            self.target_input = tf.placeholder(tf.float32, shape=(None, BOARD_SIZE), name='targets')
            target = self.target_input

            net = self.input_positions

            net = self.add_dense_layer(net, BOARD_SIZE * 3 * 9, tf.nn.relu)

            self.q_values = self.add_dense_layer(net, BOARD_SIZE, name='q_values')

            self.probabilities = tf.nn.softmax(self.q_values, name='probabilities')
            mse = tf.losses.mean_squared_error(predictions=self.q_values, labels=target)
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learningRate).minimize(mse,
                                                                                                          name='train')


class NNQPlayer(Player):

    def board_state_to_nn_input(self, state):
        res = np.array([(state == self.side).astype(int),
                        (state == Board.other_side(self.side)).astype(int),
                        (state == EMPTY).astype(int)])
        return res.reshape(-1)

    def __init__(self, name, reward_discount=0.95, win_value=1.0, draw_value=0.0,
                 loss_value=-1.0, learning_rate=0.01, training=True):
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

    def new_game(self, side):
        self.side = side
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.values_log = []

    def calculate_targets(self):
        game_length = len(self.action_log)
        targets = []

        for i in range(game_length):
            target = np.copy(self.values_log[i])

            target[self.action_log[i]] = self.reward_discount * self.next_max_log[i]
            targets.append(target)

        return targets

    def get_probs(self, input_pos):
        probs, qvalues = tfsn.get_session().run([self.nn.probabilities, self.nn.q_values],
                                                feed_dict={self.nn.input_positions: [input_pos]})
        return probs[0], qvalues[0]

    def move(self, board):
        self.board_position_log.append(board.state.copy())
        nn_input = self.board_state_to_nn_input(board.state)

        probs, qvalues = self.get_probs(nn_input)

        qvalues = np.copy(qvalues)

        for index, p in enumerate(qvalues):
            if not board.is_legal(index):
                probs[index]=-1
                # qvalues[index] = min(self.loss_value, self.draw_value, self.win_value)-1

        move = np.argmax(probs)

        if len(self.action_log) > 0:
            self.next_max_log.append(qvalues[move])

        self.values_log.append(qvalues)

        _, res, finished = board.move(move, self.side)

        self.action_log.append(move)

        return res, finished

    def final_result(self, result):
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

        self.next_max_log.append(reward)

        if self.training:
            targets = self.calculate_targets()
            nn_input = [self.board_state_to_nn_input(x) for x in self.board_position_log]
            tfsn.get_session().run([self.nn.train_step],
                                   feed_dict={self.nn.input_positions: nn_input, self.nn.target_input: targets})
