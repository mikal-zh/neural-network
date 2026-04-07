##
## EPITECH PROJECT, 2025
## G-CNA-500-PAR-5-1-neuralnetwork-4
## File description:
## FEN parsing utilities for chess inputs
##

import numpy as np

NB_PIECE = 12
POSSIBLE_CASE = 13

W_KING = 12
W_QUEEN = 11
W_ROOK = 10
W_BISHOP = 9
W_KNIGHT = 8
W_PAWN = 7

B_KING = 6
B_QUEEN = 5
B_ROOK = 4
B_BISHOP = 3
B_KNIGHT = 2
B_PAWN = 1

EMPTY = 0
EMPTY_CASE = 'e'

NOTHING = 0
W_CHECKMATE = 1
B_CHECKMATE = 2
W_CHECK = 3
B_CHECK = 4

HANDLED_CASES = 5

CHESS_SEP = '/'
DATA_SEP = ' '

CHESS_INPUTS = 832

def create_piece_vector(index: int):
    vec = np.zeros(POSSIBLE_CASE)
    vec[index] = 1
    return vec

chessboard: dict = {
    'k': create_piece_vector(B_KING),
    'K': create_piece_vector(W_KING),
    'q': create_piece_vector(B_QUEEN),
    'Q': create_piece_vector(W_QUEEN),
    'r': create_piece_vector(B_ROOK),
    'R': create_piece_vector(W_ROOK),
    'b': create_piece_vector(B_BISHOP),
    'B': create_piece_vector(W_BISHOP),
    'n': create_piece_vector(B_KNIGHT),
    'N': create_piece_vector(W_KNIGHT),
    'p': create_piece_vector(B_PAWN),
    'P': create_piece_vector(W_PAWN),
    'e': create_piece_vector(EMPTY)
}

AWAITED_OUTPUT: dict = {
    'Nothing': NOTHING,
    'Checkmate White': W_CHECKMATE,
    'Checkmate Black': B_CHECKMATE,
    'Check White': W_CHECK,
    'Check Black': B_CHECK
}

def load_inputs(line: str):
    inputs: list[float] = []
    for c in line:
        if (c == DATA_SEP):
            break
        if (c == CHESS_SEP):
            continue
        if (c.isdigit()):
            for _ in range(int(c)):
                inputs.extend(chessboard[EMPTY_CASE])
        else:
            inputs.extend(chessboard[c])
    return inputs

def get_awaited_output(line: str):
    line = line.strip()
    for key in AWAITED_OUTPUT.keys():
        if line.endswith(key):
            output: int = AWAITED_OUTPUT[key]
            return [0 if case != output else 1 for case in range(HANDLED_CASES)]
    raise KeyError(f"No valid output found in line: {line}")