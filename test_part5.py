from util import battle
from tic_tac_toe.Player import Player
from tic_tac_toe.TFSessionManager import TFSessionManager
import tensorflow as tf


def evaluate_players(p1: Player, p2: Player, games_per_battle=100, num_battles=100):
    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0


    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, False)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter = game_counter + 1
        game_number.append(game_counter)

    return game_number, p1_wins, p2_wins, draws


import matplotlib.pyplot as plt
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.EGreedyNNQPlayer import EGreedyNNQPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent

tf.reset_default_graph()

nnplayer = EGreedyNNQPlayer("QLearner1", win_value=100.0, loss_value=-100.0)
nn2player = EGreedyNNQPlayer("QLearner2", win_value=100.0, loss_value=-100.0)
# nnplayer = EGreedyNNQPlayer("QLearner1")#, learning_rate=0.001, win_value=10.0, loss_value=-10.0)
# nn2player = EGreedyNNQPlayer("QLearner2")#, learning_rate=0.001, win_value=10.0, loss_value=-10.0)
mm_player = MinMaxAgent()
rndplayer = RandomPlayer()

TFSessionManager.set_session(tf.Session())
TFSessionManager.get_session().run(tf.global_variables_initializer())

# game_number, p1_wins, p2_wins, draws = evaluate_players(rndplayer, nnplayer, num_battles=10000) #, num_battles = 20)
# game_number, p1_wins, p2_wins, draws = evaluate_players(rndplayer, nnplayer) #, num_battles = 20)
game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, nn2player, num_battles=100)  # , num_battles = 20)

game_number, p1_wins, p2_wins, draws = evaluate_players(mm_player, nn2player, num_battles=100)  # , num_battles = 20)

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
TFSessionManager.set_session(None)
