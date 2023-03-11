from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY
from util import print_board, play_game, battle
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.TabularQPlayer import TQPlayer
from tic_tac_toe.SimpleNNQPlayer import NNQPlayer
from tic_tac_toe.TFSessionManager import TFSessionManager
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

board = Board()
nnplayer = NNQPlayer("QLearner1")
nnplayer2 = NNQPlayer("QLearner2")

deep_nnplayer = NNQPlayer("DeepQLearner1")

rndplayer = RandomPlayer()
mm_player = MinMaxAgent()
tq_player = TQPlayer()

p1_wins = []
p2_wins = []
draws = []
game_number = []
game_counter = 0

num_battles = 10
games_per_battle = 100
num_training_battles = 1000

TFSessionManager.set_session(tf.compat.v1.Session())

TFSessionManager.get_session().run(tf.compat.v1.global_variables_initializer())
writer = None  # tf.summary.FileWriter('log', TFSessionManager.get_session().graph)

# nnplayer rndplayer mm_player
p1_t = deep_nnplayer
p2_t = mm_player

p1 = p1_t
p2 = p2_t

# nnplayer.training= False
# nnplayer2.training= False

for i in range(num_training_battles):
    p1win, p2win, draw = battle(p1_t, p2_t, games_per_battle, False)
    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    game_counter = game_counter + 1
    game_number.append(game_counter)

# nnplayer.training= False
# nnplayer2.training= False

for i in range(num_battles):
    p1win, p2win, draw = battle(p1, p2, games_per_battle, False)
    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    game_counter = game_counter + 1
    game_number.append(game_counter)

if writer is not None:
    writer.close()
TFSessionManager.set_session(None)

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
