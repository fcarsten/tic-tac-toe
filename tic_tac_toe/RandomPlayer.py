#
# Copyright 2018 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

from tic_tac_toe.Board import Board, GameResult
from tic_tac_toe.Player import Player


class RandomPlayer(Player):
    def __init__(self):
        self.side = None
        self.result = GameResult.NOT_FINISHED
        super().__init__()

    def move(self, board: Board) -> (GameResult, bool):
        _, res, finished = board.move(board.random_empty_spot(), self.side)
        return res, finished

    def final_result(self, result: GameResult):
        self.result = result

    def new_game(self, side: int):
        self.side = side
        self.result = GameResult.NOT_FINISHED

