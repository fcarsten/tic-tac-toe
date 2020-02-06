from util import battle
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.SimpleNNQPlayerTF2 import NNQPlayerTF2
import matplotlib.pyplot as plt
import time
import tensorflow as tf

def run_test():
    start_time = time.time()

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

#tf.compat.v1.disable_eager_execution()

# import cProfile
# cProfile.run('run_test()')

# with tf.Graph().as_default():
#     run_test()

from tensorflow.python.eager import context

with context.eager_mode():
    run_test()