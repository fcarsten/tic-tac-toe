#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from tic_tac_toe.Board import Board, EMPTY, WIN, LOSE, NEUTRAL
import random

class RndMinMaxAgent:
    cache = {}

    def __init__(self):
        self.side = None
        self.result = NEUTRAL

    def new_game(self, side):
        self.side = side
        self.result = NEUTRAL

    def final_result(self, sess, result):
        self.result = result

    def is_trainable(self):
        return False

    def _min(self, board):
        # board.print_board()
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        winner = board.check_win()
        if winner == self.side:
            return 1, -1
        elif winner == board.other_side(self.side):
            return 0, -1

        min = 0.5  # 0.5 means DRAW
        action = -1

        best_moves = {(min, action)}
        for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
            b = Board(board.state)
            b.move(index, board.other_side(self.side))

            res, _ = self._max(b)
            if res < min or action == -1:
                min = res
                action = index
                best_moves = {(min, action)}
            elif res == min:
                action = index
                best_moves.add((min, action))

        best_moves = tuple(best_moves)
        RndMinMaxAgent.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def _max(self, board):
        # board.print_board()
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        winner = board.check_win()
        if winner == self.side:
            return 1, -1
        elif winner == board.other_side(self.side):
            return 0, -1

        max = 0.5  # 0.5 means DRAW
        action = -1

        best_moves = {(max, action)}
        for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
            b = Board(board.state)
            b.move(index, self.side)

            res, _ = self._min(b)
            if res > max or action == -1:
                max = res
                action = index
                best_moves = {(max, action)}
            elif res == max:
                action = index
                best_moves.add((max, action))

        best_moves = tuple(best_moves)
        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def move(self, sess, board):
        score, action = self._max(board)
        _, res, finished = board.move(action, self.side)
        return res, finished
