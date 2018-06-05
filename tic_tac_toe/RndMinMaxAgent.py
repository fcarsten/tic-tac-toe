#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from tic_tac_toe.Board import Board, EMPTY, GameResult
from tic_tac_toe.Player import Player
import random


class RndMinMaxAgent(Player):
    """
    A computer player implementing the Min Max algorithm.

    This player behaves non-deterministically. That is, for a given board position the player if more than one move
    with the maximum evaluation exist, the player will randomly choose one of those moves.

    Already evaluated board positions are cached for efficiency.
    """

    WIN_VALUE = 1
    DRAW_VALUE = 0
    LOSS_VALUE = -1

    def __init__(self):
        """
        Getting ready for playing tic tac toe.
        """
        self.side = None

        """
        Cache to store the evaluation of board positions that we have already looked at. This avoids repeating a lot
        of work as we do not look at all the possible continuation from this position again.
        """
        self.cache = {}

        super().__init__()

    def new_game(self, side: int):
        """
        Setting the side for the game to come. Noting else to do.
        :param side: The side this player will be playing
        """
        if self.side != side:
            self.side = side
            self.cache = {}

    def final_result(self, result: GameResult):
        """
        Does nothing.
        :param result: The result of the game that just finished
        :return:
        """
        pass

    def _min(self, board: Board) -> int:
        """
        Evaluate the board position `board` from the Minimizing player's point of view.

        :param board: The board position to evaluate
        :return: returns the best Move in this situation. Returns -1 for best move if the game has already
        finished
        """

        #
        # First we check if we have seen this board position before, and if yes just return a random choice
        # from the cached values
        #
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        #
        # If the game has already finished we return. Otherwise we look at possible continuations
        #
        winner = board.who_won()
        if winner == self.side:
            best_moves = {(self.WIN_VALUE, -1)}
        elif winner == board.other_side(self.side):
            best_moves = {(self.LOSS_VALUE, -1)}
        else:
            #
            # Init the min value as well as action. Min value is set to DRAW as this value will pass through in case
            # of a draw
            #
            min_value = self.DRAW_VALUE
            action = -1
            best_moves = {(min_value, action)}
            for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
                b = Board(board.state)
                b.move(index, board.other_side(self.side))

                res, _ = self._max(b)
                if res < min_value or action == -1:
                    min_value = res
                    action = index
                    best_moves = {(min_value, action)}
                elif res == min_value:
                    action = index
                    best_moves.add((min_value, action))

        best_moves = tuple(best_moves)
        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def _max(self, board: Board) -> int:
        """
        Evaluate the board position `board` from the Maximizing player's point of view.

        :param board: The board position to evaluate
        :return: Best Move in this situation. Returns -1 for best move if the game has already finished
        """

        #
        # First we check if we have seen this board position before, and if yes just return a random choice
        # from the cached values
        #
        board_hash = board.hash_value()
        if board_hash in self.cache:
            return random.choice(self.cache[board_hash])

        #
        # If the game has already finished we return. Otherwise we look at possible continuations
        #
        winner = board.who_won()
        if winner == self.side:
            best_moves = {(self.WIN_VALUE, -1)}
        elif winner == board.other_side(self.side):
            best_moves = {(self.LOSS_VALUE, -1)}
        else:
            #
            # Init the min value as well as action. Min value is set to DRAW as this value will pass through in case
            # of a draw
            #
            max_value = self.DRAW_VALUE
            action = -1
            best_moves = {(max_value, action)}
            for index in [i for i, e in enumerate(board.state) if board.state[i] == EMPTY]:
                b = Board(board.state)
                b.move(index, self.side)

                res, _ = self._min(b)
                if res > max_value or action == -1:
                    max_value = res
                    action = index
                    best_moves = {(max_value, action)}
                elif res == max_value:
                    action = index
                    best_moves.add((max_value, action))

        best_moves = tuple(best_moves)
        self.cache[board_hash] = best_moves

        return random.choice(best_moves)

    def move(self, board: Board) -> (GameResult, bool):
        """

        Making a move according to the MinMax algorithm. If more than one best move exist, chooses amongst them
        randomly.

        :param board: The board to make a move on
        :return: The result of the move
        """
        score, action = self._max(board)
        _, res, finished = board.move(action, self.side)
        return res, finished
