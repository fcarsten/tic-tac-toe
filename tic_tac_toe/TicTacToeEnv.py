#
#
# gym TicTacToe Env
#
# based on https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent/
#
import logging

import gym
import numpy as np
from gym import spaces
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from tic_tac_toe import Player, RandomPlayer
from tic_tac_toe.Board import Board, GameResult, BOARD_SIZE, CROSS, NAUGHT, EMPTY, BOARD_DIM


def valid_actions_mask(obs):
    return obs, np.copy(obs[2]).reshape(-1)


class TicTacToeEnv(py_environment.PyEnvironment):

    metadata = {'render.modes': ['human']}

    def get_info(self):
        return None

    def __init__(self, player: Player, env_moves_first: bool):
        self.player = player
        self.env_moves_first = env_moves_first
        self.board = Board()
        self.done = False

        if self.env_moves_first:
            self.other_side = NAUGHT
        else:
            self.other_side = CROSS

        self.reward = None
        self.reset()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=BOARD_SIZE-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(3, BOARD_DIM, BOARD_DIM), dtype=np.int32, minimum=0, maximum=1, name='observation')

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    # def set_start_mark(self, mark):
    #     self.start_mark = mark

    def _reset(self):
        self.board.reset()
        self.reward = 0

        if self.env_moves_first:
            self.player.new_game(CROSS)
            self.player.move(self.board)
        else:
            self.player.new_game(NAUGHT)

        # self.mark = self.start_mark
        self.done = False
        return ts.restart(self._get_obs())

    def _step(self, action):
        """Step environment by action.
        Args:
            action (int): Location
        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        if self.done:
            return self._reset()

        _, result, self.done = self.board.move(action, self.other_side)

        if self.done:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, self.done = self.player.move(self.board)
            if self.done:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

        if self.done:
            self.player.final_result(final_result)
            if final_result == GameResult.DRAW:
                self.reward = 0
            elif (final_result == GameResult.NAUGHT_WIN and self.other_side == NAUGHT) or \
                    (final_result == GameResult.CROSS_WIN and self.other_side == CROSS):
                self.reward = 1
            else:
                self.reward = -1
            return ts.termination(self._get_obs(), self.reward)

        return ts.transition(self._get_obs(), 0)

    def _get_obs(self):
        state = self.board.state
        res = np.array([(state == self.other_side).astype(int),
                        (state == Board.other_side(self.other_side)).astype(int),
                        (state == EMPTY).astype(int)], dtype=np.int32)
        res = res.reshape(3, BOARD_DIM, BOARD_DIM)
        return res;
        # return tuple(self.board), self.mark

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self.board.print_board()
            print('')
        else:
            logging.info(self.board)
            logging.info('')

    def show_episode(self, human, episode):
        self._show_episode(print if human else logging.warning, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    # def _show_board(self, showfn):
    #     """Draw tictactoe board."""
    #     for j in range(0, 9, 3):
    #         def mark(i):
    #             return tomark(self.board[i]) if not self.show_number or \
    #                                             self.board[i] != 0 else str(i + 1)
    #
    #         showfn(LEFT_PAD + '|'.join([mark(i) for i in range(j, j + 3)]))
    #         if j < 6:
    #             showfn(LEFT_PAD + '-----')

    def show_turn(self, human, mark):
        self._show_turn(print if human else logging.info, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print if human else logging.info, mark, reward)

    def _show_result(self, showfn, mark, reward):
        showfn("==== TODO: Implement show_result ====")
        # status = check_game_status(self.board)
        # assert status >= 0
        # if status == 0:
        #     showfn("==== Finished: Draw ====")
        # else:
        #     msg = "Winner is '{}'!".format(tomark(status))
        #     showfn("==== Finished: {} ====".format(msg))
        # showfn('')

    def available_actions(self):
        return [i for i, c in enumerate(self.board.state) if c == 0]
