import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import evaluate_players
from tic_tac_toe.TFSessionManager import TFSessionManager

from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
from tic_tac_toe.DirectPolicyAgent import DirectPolicyAgent

min_reward = -3
max_reward = 3

num_reward_steps = 1+ max_reward - min_reward

rewards = np.zeros( (num_reward_steps, num_reward_steps) )


for loss_reward in range(min_reward, max_reward):
    for draw_reward in range(loss_reward + 1, max_reward + 1):

        tf.reset_default_graph()
        TFSessionManager.set_session(tf.Session())

        sess = TFSessionManager.get_session()

        nnplayer = DirectPolicyAgent("PolicyLearner1", loss_value= loss_reward, draw_value= draw_reward)
        rm_player = RndMinMaxAgent()


        sess.run(tf.global_variables_initializer())

        game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, rm_player, num_battles=1000, silent=True)  # , num_battles = 20)

        print("With loss reward {} and draw reward {} we get draws: {}".format(
            loss_reward, draw_reward, draws[-1]))

        rewards[loss_reward-min_reward, draw_reward-min_reward] = draws[-1]


        TFSessionManager.set_session(None)





fig, ax = plt.subplots()
im = ax.imshow(rewards)

reward_range = np.arange(num_reward_steps+1)

# We want to show all ticks...
ax.set_xticks(reward_range)
ax.set_yticks(reward_range)
# ... and label them with the respective list entries

reward_labels = np.arange(start= min_reward, stop=max_reward + 2)

ax.set_xticklabels(reward_labels)
ax.set_yticklabels(reward_labels)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(num_reward_steps):
    for j in range(num_reward_steps):
        text = ax.text(j, i, rewards[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Test Title")
fig.tight_layout()
plt.show()