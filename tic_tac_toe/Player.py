#
# Copyright 2018 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
from abc import ABC, abstractmethod

from tic_tac_toe.Board import Board, GameResult


class Player(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def move(self, board: Board) -> (GameResult, bool):
        pass

    @abstractmethod
    def final_result(self, result: GameResult):
        pass

    @abstractmethod
    def new_game(self, side: int):
        pass
