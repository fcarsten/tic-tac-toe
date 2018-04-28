#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from tic_tac_toe.Board import Board, EMPTY, GameResult
from tic_tac_toe.Player import Player


class MinMaxAgent(Player):
    cache = {}

    def __init__(self):
        self.side = None
        self.result = None

    def new_game(self, side):
        self.side = side
        self.result = GameResult.NOT_FINISHED

    def final_result(self, sess, result):
        self.result = result

    def is_trainable(self):
        return False

    def _min(self, board):
        # board.print_board()
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return self.cache[board_hash]

        winner = board.check_win()
        if winner == self.side:
            return 1, -1
        elif winner == board.other_side(self.side):
            return 0, -1

        min = 0.5  # 0.5 means DRAW
        action = -1

        for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
            b = Board(board.state)
            b.move(index, board.other_side(self.side))

            res, _ = self._max(b)
            if res < min or action == -1:
                min = res
                action = index
                if min == 0:
                    self.cache[board_hash] = (min, action)
                    return min, action

        self.cache[board_hash] = (min, action)
        return min, action

    def _max(self, board):
        # board.print_board()
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return self.cache[board_hash]

        winner = board.check_win()
        if winner == self.side:
            return 1, -1
        elif winner == board.other_side(self.side):
            return 0, -1

        max = 0.5  # 0.5 means DRAW
        action = -1

        for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
            b = Board(board.state)
            b.move(index, self.side)

            res, _ = self._min(b)
            if res > max or action == -1:
                max = res
                action = index
                if max == 1:
                    self.cache[board_hash] = (max, action)
                    return max, action

        self.cache[board_hash] = (max, action)
        return max, action

    def move(self, sess, board):
        score, action = self._max(board)
        _, res, finished = board.move(action, self.side)
        return res, finished
