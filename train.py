# train.py
import argparse
from dataset_loader import pgn_to_dataset
from model import build_model, save_model
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', type=str, default='data/sample_games.pgn')
    parser.add_argument('--out', type=str, default='models/chess_eval.h5')
    parser.add_argument('--max_games', type=int, default=2000)
    args = parser.parse_args()

    print("Loading PGN and creating dataset (this may take a while)...")
    X, y = pgn_to_dataset(args.pgn, max_games=args.max_games)
    print("Dataset:", X.shape, y.shape)
    # shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    input_dim = X.shape[1]
    model = build_model(input_dim)
    model.summary()
    # train-test split
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=10, batch_size=128)
    save_model(model, args.out)
    print("Saved model to", args.out)

if __name__ == "__main__":
    main()
