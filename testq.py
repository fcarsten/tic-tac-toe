from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT, EMPTY
from util import print_board, play_game, battle
from tic_tac_toe.RandomPlayer import RandomPlayer
from tic_tac_toe.MinMaxAgent import MinMaxAgent
from tic_tac_toe.RndMinMaxAgent import RndMinMaxAgent
from tic_tac_toe.TabularQPlayer import TQPlayer
import matplotlib.pyplot as plt

board = Board()
player1 = TQPlayer()
player2 = MinMaxAgent()


p1_wins = []
p2_wins = []
draws = []
count = []
num_battles = 100
games_per_battle = 100

for i in range(num_battles):
    p1win, p2win, draw = battle(player1, player2, games_per_battle, True)
    p1_wins.append(p1win)
    p2_wins.append(p2win)
    draws.append(draw)
    count.append(i*games_per_battle)

p = plt.plot(count, draws, 'r-', count, p1_wins, 'g-', count, p2_wins, 'b-')

plt.show()
