#
# Copyright 2018 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#

import numpy as np
from enum import Enum


#
# class to encode different states of the game. A game can be in progress (NOT_FINISHED), lost, won, or draw
#
class GameResult(Enum):
    NOT_FINISHED = 0
    NAUGHT_WIN = 1
    CROSS_WIN = 2
    DRAW = 3


#
# Values to encode the current content of a field on the board. A field can be empty, contain a naught, or
# contain a cross
#
EMPTY = 0
NAUGHT = 1
CROSS = 2

#
# Define the length and width of the board. Has to be 3 at the moment, or parts of the code will break. Also,
# the game mechanics kind of require this dimension unless other rules are changed as well. Encoding as a variable
# to make the code more readable
#
BOARD_DIM = 3  # type: int
BOARD_SIZE = BOARD_DIM * BOARD_DIM


#
# The class to encode a tic-tac-toe board, including its current state of pieces.
# Also contains various utility methods.
#
class Board:
    #
    # We will use these starting positions and directions when checking if a move resulted in the game being
    # won by one of the sides.
    #
    WIN_CHECK_DIRS = {0: [(1, 1), (1, 0), (0, 1)],
                      1: [(1, 0)],
                      2: [(1, 0), (1, -1)],
                      3: [(0, 1)],
                      6: [(0, 1)]}

    #
    # Encode the current state of the game (board positions) as an integer. Will be used for caching evaluations
    #
    def hash_value(self) -> int:
        res = 0
        for i in range(BOARD_SIZE):
            res *= 3
            res += self.state[i]

        return res

    #
    # Utility method to return the value of the other player than the one passed as input
    #
    @staticmethod
    def other_side(side: int) -> int:
        if side == EMPTY:
            raise ValueError("EMPTY has no 'other side'")

        if side == CROSS:
            return NAUGHT

        if side == NAUGHT:
            return CROSS

        raise ValueError("{} is not a valid side".format(side))

    #
    # Create a new Board. If a state is passed in, we use that otherwise we initialize with an empty board
    #
    def __init__(self, s=None):
        if s is None:
            self.state = np.ndarray(shape=(1, BOARD_SIZE), dtype=int)[0]
            self.reset()
        else:
            self.state = s.copy()

    #
    # Converts a 2D board position to a 1D board position.
    # Various pieces of code prefer one over the other.
    #
    def coord_to_pos(self, coord: (int, int)) -> int:
        return coord[0] * BOARD_DIM + coord[1]

    #
    # Converts a 1D board position to a 2D board position.
    # Various pieces of code prefer one over the other.
    #
    def pos_to_coord(self, pos: int) -> (int, int):
        return pos // BOARD_DIM, pos % BOARD_DIM

    #
    # Resets the game board. All fields are set to be EMPTY.
    #
    def reset(self):
        self.state.fill(EMPTY)

    #
    # Counts and returns the number of empty fields on the board.
    #
    def num_empty(self) -> int:
        return np.count_nonzero(self.state == EMPTY)

    #
    # Returns a random empty spot on the board.
    #
    def random_empty_spot(self) -> int:
        index = np.random.randint(self.num_empty())
        for i in range(9):
            if self.state[i] == EMPTY:
                if index == 0:
                    return i
                else:
                    index = index - 1

    #
    # Tests whether a board position can be played, i.e. is currently empty
    #
    def is_legal(self, pos: int) -> bool:
        return self.state[pos] == EMPTY

    #
    # Places a piece of side "side" at position "position". The position is to be provided as 1D.
    # Throws a ValueError if the position is not EMPTY
    # returns the new state of the board, the game result after this move, and whether this move has finished the game
    #
    def move(self, position: int, side: int) -> (np.ndarray, GameResult, bool):
        if self.state[position] != EMPTY:
            print('Illegal move')
            raise ValueError("Invalid move")

        self.state[position] = side

        if self.check_win():
            return self.state, GameResult.CROSS_WIN if side == CROSS else GameResult.NAUGHT_WIN, True

        if self.num_empty() == 0:
            return self.state, GameResult.DRAW, True

        return self.state, GameResult.NOT_FINISHED, False

    #
    # Applies 2D direction dir to 1D position pos.
    # Returns the resulting 1D position, or -1 if the resulting position would not be a valid board position.
    #
    def apply_dir(self, pos: int, direction: (int, int)) -> int:
        row = pos // 3
        col = pos % 3
        row += direction[0]
        if row < 0 or row > 2:
            return -1
        col += direction[1]
        if col < 0 or col > 2:
            return -1

        return row * 3 + col

    #
    # Checks and returns whether there are 3 pieces of the same side in a row if following direction dir
    #
    def check_win_in_dir(self, pos: int, direction: (int, int)) -> bool:
        c = self.state[pos]
        if c == EMPTY:
            return False

        p1 = int(self.apply_dir(pos, direction))
        p2 = int(self.apply_dir(p1, direction))

        if p1 == -1 or p2 == -1:
            return False

        if c == self.state[p1] and c == self.state[p2]:
            return True

        return False

    #
    # Returns whether any side has won the game
    #
    def check_win(self) -> bool:
        for start_pos in self.WIN_CHECK_DIRS:
            if self.state[start_pos] != EMPTY:
                for direction in self.WIN_CHECK_DIRS[start_pos]:
                    res = self.check_win_in_dir(start_pos, direction)
                    if res:
                        return True

        return False

    #
    # Return 'x', 'o', or ' ' depending on what piece is on 1D position pos
    #
    def state_to_char(self, pos, html=False):
        if (self.state[pos]) == EMPTY:
            return '&ensp;' if html else ' '

        if (self.state[pos]) == NAUGHT:
            return 'o'

        return 'x'


    def html_str(self) -> str:
        data = self.state_to_charlist(True)
        html = '<table border="1"><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
        return html

    def state_to_charlist(self, html=False):
        res = []
        for i in range(3):
            line = [self.state_to_char(i * 3, html) ,
                    self.state_to_char(i * 3 + 1, html),
                    self.state_to_char(i * 3 + 2, html)]
            res.append(line)

        return res

    #
    # Print an ASCII representation of the board
    #
    def print_board(self):
        for i in range(3):
            board_str = self.state_to_char(i * 3) + '|' + self.state_to_char(i * 3 + 1) \
                        + '|' + self.state_to_char(i * 3 + 2)

            print(board_str)
            if i != 2:
                print("-----")

        print("")
