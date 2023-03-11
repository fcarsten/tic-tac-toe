import tensorflow as tf
import matplotlib.pyplot as plt

from util import evaluate_players
from tic_tac_toe.TFSessionManager import TFSessionManager
from tic_tac_toe.RandomPlayer import RandomPlayer

from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
from tic_tac_toe.DirectPolicyAgent import DirectPolicyAgent

TENSORLOG_DIR = './graphs'

if tf.compat.v1.gfile.Exists(TENSORLOG_DIR):
    tf.compat.v1.gfile.DeleteRecursively(TENSORLOG_DIR)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

nnplayer = DirectPolicyAgent("PolicyLearner1")
# nn2player = EGreedyNNQPlayer("QLearner2", win_value=100.0, loss_value=-100.0)
# nnplayer = EGreedyNNQPlayer("QLearner1")#, learning_rate=0.001, win_value=10.0, loss_value=-10.0)
# nn2player = EGreedyNNQPlayer("QLearner2")#, learning_rate=0.001, win_value=10.0, loss_value=-10.0)
mm_player = MinMaxAgent()
rndplayer = RandomPlayer()
rm_player = RndMinMaxAgent()

TFSessionManager.set_session(tf.compat.v1.Session())

sess = TFSessionManager.get_session()
writer = tf.compat.v1.summary.FileWriter(TENSORLOG_DIR, sess.graph)
nnplayer.writer = writer

sess.run(tf.compat.v1.global_variables_initializer())

# game_number, p1_wins, p2_wins, draws = evaluate_players(rndplayer, nnplayer, num_battles=10000) #, num_battles = 20)
# game_number, p1_wins, p2_wins, draws = evaluate_players(rndplayer, nnplayer) #, num_battles = 20)
# game_number, p1_wins, p2_wins, draws = evaluate_players( mm_player, nnplayer, num_battles=300)  # , num_battles = 20)
game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, rm_player, num_battles=1000, writer=writer)  # , num_battles = 20)
# game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, rndplayer, num_battles=100)  # , num_battles = 20)

# game_number, p1_wins, p2_wins, draws = evaluate_players(mm_player, nn2player, num_battles=100)  # , num_battles = 20)
writer.close()

p = plt.plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')

plt.show()
TFSessionManager.set_session(None)
