import tkinter as tk
from tkinter import messagebox
import chess
import chess.engine
import chess.svg
from PIL import Image, ImageTk
import os
from ai_player import NN_Evaluator, choose_move  # your AI evaluator
import tensorflow as tf

# Load your trained model
model_path = "model/chess_eval.keras"
evaluator = NN_Evaluator(model_path)

# Chess board setup
board = chess.Board()
selected_square = None

# GUI setup
root = tk.Tk()
root.title("Chess AI")
root.resizable(False, False)

# Load images
pieces = {}
for color in ['w', 'b']:
    for piece in ['p','n','b','r','q','k']:
        path = f"images/{color}{piece}.png"
        pieces[f"{color}{piece}"] = ImageTk.PhotoImage(Image.open(path).resize((64,64)))

# Canvas
canvas = tk.Canvas(root, width=512, height=512)
canvas.pack()

SQUARE_COLORS = ["#F0D9B5", "#B58863"]

def draw_board():
    canvas.delete("all")
    for row in range(8):
        for col in range(8):
            x1, y1 = col*64, row*64
            x2, y2 = x1+64, y1+64
            color = SQUARE_COLORS[(row+col)%2]
            canvas.create_rectangle(x1, y1, x2, y2, fill=color)
            
            square = chess.square(col, 7-row)  # chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                key = ('w' if piece.color else 'b') + piece.symbol().lower()
                canvas.create_image(x1, y1, anchor="nw", image=pieces[key])

def click(event):
    global selected_square
    col = event.x // 64
    row = 7 - (event.y // 64)
    square = chess.square(col, row)

    global board
    if selected_square is None:
        # select piece
        piece = board.piece_at(square)
        if piece and piece.color == chess.WHITE:  # human plays white
            selected_square = square
    else:
        move = chess.Move(selected_square, square)
        if move in board.legal_moves:
            board.push(move)
            draw_board()
            root.update()
            selected_square = None
            root.after(100, ai_move)
        else:
            selected_square = None

def ai_move():
    global board
    if board.is_game_over():
        messagebox.showinfo("Game Over", f"Result: {board.result()}")
        return

    # best_move = evaluator.get_best_move(board, depth=2)
    best_move = choose_move(board, evaluator, depth=2)
    board.push(best_move)
    draw_board()
    if board.is_game_over():
        messagebox.showinfo("Game Over", f"Result: {board.result()}")

canvas.bind("<Button-1>", click)
draw_board()
root.mainloop()
