from util import battle
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent

# battle(RandomPlayer(), RndMinMaxAgent())
import tensorflow as tf
from tensorboard import main as tb

tf.flags.FLAGS.logdir = "./graphs"
tb.main()
