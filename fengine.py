import chess
import chess.engine
import sys

import numpy as np

X_train = []
Y_train = []

fen_num = int(sys.argv[1])
fens = np.load("fen_list"+str(fen_num)+".npy")
engine = chess.engine.SimpleEngine.popen_uci("./stockfish.exe")
i=0
for f in fens:
    i+=1
    if i %1000 ==0:
        print(i)
    b=chess.Board(fen = f)
    Y_train.append(max(min(engine.analyse(b, chess.engine.Limit(depth=12))['score'].white().score(mate_score=1500), 1500), -1500))


engine.quit()

Y_train = np.array(Y_train)

np.save("y_12fen"+str(fen_num)+".npy", Y_train)
