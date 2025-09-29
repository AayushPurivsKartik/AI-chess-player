# utils.py
import numpy as np
import chess

PIECE_TO_PLANE = {
    chess.PAWN:0,
    chess.KNIGHT:1,
    chess.BISHOP:2,
    chess.ROOK:3,
    chess.QUEEN:4,
    chess.KING:5
}

def board_to_tensor(board: chess.Board):
    """
    Encode board as 8x8x12 numpy float32 array (planes: white pieces then black pieces).
    Also return side_to_move scalar (1 for white to move, 0 for black).
    """
    tensor = np.zeros((8,8,12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        r = 7 - (square // 8)  # row 0 is top (rank 8)
        c = square % 8
        plane = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.WHITE:
            tensor[r, c, plane] = 1.0
        else:
            tensor[r, c, 6 + plane] = 1.0
    stm = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor, stm

def tensor_to_flat(tensor, stm):
    """Flatten to 8*8*12 + 1 for feeding into a dense network."""
    return np.concatenate([tensor.ravel(), np.array([stm], dtype=np.float32)])
