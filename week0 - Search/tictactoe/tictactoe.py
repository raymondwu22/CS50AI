"""
Tic Tac Toe Player
"""

import math
import copy
from collections import Counter

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if terminal(board):
        return

    cnt = Counter()
    for row in board:
        for item in row:
            cnt[item] += 1
    if cnt[EMPTY] == 9 or cnt[X] == cnt[O]:
        return X
    elif cnt[X] > cnt[O]:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    available = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                available.append((i, j))

    return available


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    (row, col) = action
    if board[row][col] != EMPTY:
        raise ValueError('The spot is taken already!')

    current_player = player(board)
    copy_board = copy.deepcopy(board)
    copy_board[row][col] = current_player
    return copy_board


# helper function to detect a row victory
def rowWin(board):
    for row in range(3):
        if board[row][0] == X and board[row][1] == X and board[row][2] == X:
            return X
        elif board[row][0] == O and board[row][1] == O and board[row][2] == O:
            return O
    return None

# helper function to detect a column victory
def colWin(board):
    for col in range(3):
        if board[0][col] == X and board[1][col] == X \
                and board[2][col] == X:
            return X
        elif board[0][col] == O and board[1][col] == O \
                and board[2][col] == O:
            return O
    return None

# helper function to detect a diagonal victory
def diagonalWin(board):
    if (board[0][0] == X and board[1][1] == X and board[2][2] == X) \
            or (board[2][0] == X and board[1][1] == X and board[0][2] == X):
        return X
    elif (board[0][0] == O and board[1][1] == O and board[2][2] == O) \
            or (board[2][0] == O and board[1][1] == O and board[0][2] == O):
        return O
    return None


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if (rowWin(board) == X) or (colWin(board) == X) \
            or (diagonalWin(board) == X):
        return X
    elif (rowWin(board) == O) or (colWin(board) == O) \
            or (diagonalWin(board) == O):
        return O


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def max_action(board):
    if terminal(board):
        return utility(board)
    if board == initial_state():
        potential = actions(board)
        return (1, potential[-1])

    max_v = -math.inf
    best_action = None
    potential = actions(board)
    for action in potential:
        v = min_action(result(board, action))
        if isinstance(v, tuple):
            v = v[0]
        if v > max_v:
            max_v = v
            best_action = action
    return (max_v, best_action)


def min_action(board):
    if terminal(board):
        return utility(board)
    if board == initial_state():
        potential = actions(board)
        return (1, potential[-1])

    min_v = math.inf
    best_action = None
    potential = actions(board)
    for action in potential:
        v = max_action(result(board, action))
        if isinstance(v, tuple):
            v = v[0]
        if v < min_v:
            min_v = v
            best_action = action
    return (min_v, best_action)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if player(board) == X:
        return max_action(board)[1]
    else:
        return min_action(board)[1]
