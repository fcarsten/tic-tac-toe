from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY
from util import print_board, play_game, battle
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.TabularQPlayer import TQPlayer
from tic_tac_toe.SimpleNNQPlayerTF2 import NNQPlayerTF2
from tic_tac_toe.TFSessionManager import TFSessionManager
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
#with tf.Graph().as_default():

import time
start_time = time.time()

board = Board()
nnplayer = NNQPlayerTF2("QLearner1")

rndplayer2 = RandomPlayer()

p1_wins = []
p2_wins = []
draws = []
game_number = []
game_counter = 0

num_battles = 10
games_per_battle = 100

p1 = nnplayer
p2 = rndplayer2

for i in range(num_battles):
    p1win, p2win, draw = battle(p1, p2, games_per_battle, False)
    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    game_counter = game_counter + 1
    game_number.append(game_counter)

print("--- %s seconds ---" % (time.time() - start_time))

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
