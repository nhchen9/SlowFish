import chess
import chess.pgn
import numpy as np
import sys
pgn = open("lichess.pgn")

i = 0

n=int(sys.argv[1])

fens = []
skips = 0
while i < 100000:
    skips+=1
    if skips < 100000 * n:
        continue
    game = chess.pgn.read_game(pgn)
    board = chess.Board()
    moves =0
    for move in game.mainline_moves():
        moves+=1
        board.push(move)
        if np.random.random() < .002 * moves:
            fens.append(board.fen())
            i+=1
            if i%1000==0:
                print(i)

np.save("fen_list"+str(n)+".npy", np.array(fens))
