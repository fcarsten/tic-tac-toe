#
# Copyright 2018 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
from abc import ABC, abstractmethod

from tic_tac_toe.Board import Board, GameResult


class Player(ABC):
    """
    Abstract class defining the interface we expect any Tic Tac Toe player class to implement.
    This will allow us to pit various different implementation against each other
    """

    def __init__(self):
        """
        Nothing to do here apart from calling our super class
        """
        super().__init__()

    @abstractmethod
    def move(self, board: Board) -> (GameResult, bool):
        """
        The player should make a move on board `board` and return the result. The return result can usually
        be passed on from the corresponding method in the board class.
        :param board: The board to make a move on
        :return: The GameResult after this move, Flag to indicate whether the move finished the game
        """
        pass

    @abstractmethod
    def final_result(self, result: GameResult):
        """
        This method will be called after the game has finished. It allows the player to ponder its game move choices
        and learn from the experience
        :param result: The result of the game
        """
        pass

    @abstractmethod
    def new_game(self, side: int):
        """
        This method will be called before a game starts. It allows the players to get ready for the game and also tells
        which side it is on.
        :param side: The side the player will play in the new game
        """
        pass
