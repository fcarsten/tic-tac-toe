from tic_tac_toe.Board import Board
from util import battle
from tic_tac_toe.Player import Player
from tic_tac_toe.TFSessionManager import TFSessionManager
import tensorflow as tf


def evaluate_players(p1: Player, p2: Player, games_per_battle=100, num_battles=100):
    board = Board()

    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    TFSessionManager.set_session(tf.compat.v1.Session())
    TFSessionManager.get_session().run(tf.compat.v1.global_variables_initializer())

    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, False)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter = game_counter + 1
        game_number.append(game_counter)

    TFSessionManager.set_session(None)
    return game_number, p1_wins, p2_wins, draws


import matplotlib.pyplot as plt
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.SimpleNNQPlayer import NNQPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

nnplayer = NNQPlayer("QLearner1", learning_rate=0.01, win_value=100.0, loss_value=-100.0)
mm_player = MinMaxAgent()
rndplayer = RandomPlayer()

game_number, p1_wins, p2_wins, draws = evaluate_players(mm_player, nnplayer, num_battles=10000)  # , num_battles = 20)
# game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, mm_player) #, num_battles = 20)

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
