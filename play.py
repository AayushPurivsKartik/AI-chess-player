# play.py
import chess
from ai_player import NN_Evaluator, choose_move
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/chess_eval.h5')
    parser.add_argument('--depth', type=int, default=2)
    args = parser.parse_args()

    evaluator = NN_Evaluator(args.model)
    board = chess.Board()
    print(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # human plays white (you can swap)
            move = input("Your move (in UCI or SAN): ")
            try:
                # try SAN then UCI
                try:
                    mv = board.parse_san(move)
                except:
                    mv = chess.Move.from_uci(move)
                if mv not in board.legal_moves:
                    print("Illegal move")
                    continue
                board.push(mv)
            except Exception as e:
                print("Error:", e)
                continue
        else:
            print("AI is thinking...")
            mv = choose_move(board, evaluator, depth=args.depth)
            print("AI plays:", mv)
            board.push(mv)
        print(board)
        print("FEN:", board.fen())
    print("Game over:", board.result())

if __name__ == "__main__":
    main()
