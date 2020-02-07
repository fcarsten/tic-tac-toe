import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import tf_agents as tf_agents
import numpy as np

from tic_tac_toe.TicTacToeEnv import TicTacToeEnv, valid_actions_mask
from tic_tac_toe.RandomPlayer import RandomPlayer

tf.compat.v1.enable_v2_behavior()

print(tf.version.VERSION)

rndplayer = RandomPlayer()

ttt_env = TicTacToeEnv(rndplayer, False)


tf_agents.environments.utils.validate_py_environment(ttt_env,
                                                     observation_and_action_constraint_splitter=valid_actions_mask)

ttt_env.reset()

print('Observation Spec:')
print(ttt_env.time_step_spec().observation)

print('Reward Spec:')
print(ttt_env.time_step_spec().reward)

print('Action Spec:')
print(ttt_env.action_spec())

print('Finished')

