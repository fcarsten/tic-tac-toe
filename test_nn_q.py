from IPython.display import HTML, display
from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY
from util import print_board, play_game, battle
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
from tic_tac_toe.TabularQPlayer import TQPlayer
from tic_tac_toe.NeuralNetworkQPlayer import NNQPlayer
from tic_tac_toe.TFSessionManager import TFSessionManager
import matplotlib.pyplot as plt
import tensorflow as tf

board = Board()
player2 = NNQPlayer("QLearner1")
player1 = RandomPlayer()

p1_wins = []
p2_wins = []
draws = []
count = []
num_battles = 100
games_per_battle = 1000

TFSessionManager.set_session(tf.InteractiveSession())

TFSessionManager.get_session().run(tf.global_variables_initializer())

for i in range(num_battles):
    p1win, p2win, draw = battle(player1, player2, games_per_battle, False)
    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    count.append(i*games_per_battle)

TFSessionManager.set_session(None)

p = plt.plot(count, draws, 'r-', count, p1_wins, 'g-', count, p2_wins, 'b-')

plt.show()
