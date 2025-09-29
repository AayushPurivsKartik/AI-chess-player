# dataset_loader.py
import chess.pgn
import numpy as np
from tqdm import tqdm
from utils import board_to_tensor, tensor_to_flat

def pgn_to_dataset(pgn_path, max_games=None, min_moves=10):
    """
    Parse PGN and return X (N x input_dim) and y (N,) where y is final game result
    from the perspective of the side to move at that position.
    """
    X = []
    y = []
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1
            if max_games and game_count > max_games:
                break
            # extract result
            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            if result == "1-0":
                game_result = 1.0
            elif result == "0-1":
                game_result = 0.0
            else:
                game_result = 0.5

            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < min_moves:
                continue

            # For each move (or sample every k moves), record position before the move
            for i, move in enumerate(moves):
                # push position state (before the move)
                tensor, stm = board_to_tensor(board)
                flat = tensor_to_flat(tensor, stm)
                # label: result from perspective of side to move
                # If side to move is white and game_result==1 => label 1; if black to move invert.
                label = game_result if board.turn == chess.WHITE else (1.0 - game_result)
                X.append(flat)
                y.append(label)
                board.push(move)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

if __name__ == "__main__":
    # small test using data/sample_games.pgn
    X, y = pgn_to_dataset('data/sample_games.pgn', max_games=50)
    print("X", X.shape, "y", y.shape)
