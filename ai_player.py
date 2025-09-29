# ai_player.py
import chess
import numpy as np
from model import load_model
from utils import board_to_tensor, tensor_to_flat
import math
import time

class NN_Evaluator:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def evaluate(self, board: chess.Board):
        tensor, stm = board_to_tensor(board)
        flat = tensor_to_flat(tensor, stm).reshape(1, -1)
        pred = self.model.predict(flat, verbose=0)[0,0]  # probability side-to-move wins
        # Convert probability into a centipawn-like score for search (-1 .. 1 -> -1000 .. +1000)
        score = (pred - 0.5) * 2000.0
        # If it is black to move, model predicts black's win probability in our label scheme (we inverted),
        # but our pred is always "probability side-to-move wins" already, so we can use score directly.
        return score

def negamax(board, depth, alpha, beta, evaluator: NN_Evaluator):
    if depth == 0 or board.is_game_over():
        if board.is_game_over():
            if board.is_checkmate():
                # checkmate for side not to move (because board.turn is player to move)
                # If side to move has no moves and is in check => checkmate => side to move loses
                return -10000.0
            else:
                return 0.0
        return evaluator.evaluate(board)

    max_eval = -math.inf
    for move in board.legal_moves:
        board.push(move)
        val = -negamax(board, depth-1, -beta, -alpha, evaluator)
        board.pop()
        if val > max_eval:
            max_eval = val
        if max_eval > alpha:
            alpha = max_eval
        if alpha >= beta:
            break
    return max_eval

def choose_move(board: chess.Board, evaluator: NN_Evaluator, depth=2):
    best_move = None
    best_val = -math.inf
    alpha = -1e9
    beta = 1e9
    for move in board.legal_moves:
        board.push(move)
        val = -negamax(board, depth-1, -beta, -alpha, evaluator)
        board.pop()
        if val > best_val:
            best_val = val
            best_move = move
        if val > alpha:
            alpha = val
    return best_move
