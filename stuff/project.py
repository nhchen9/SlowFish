
import chess
import chess.pgn
import chess.engine
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import pickle

class binaryChessNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self):

        super(binaryChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 8, 3, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1, bias = True)
        self.lin = nn.Linear(128, 2)

        print(self.conv1.weight.shape)
        print(self.conv1.bias.shape)

        self.conv1.weight = nn.Parameter(torch.rand(8, 13, 3,3)-.5)
        self.conv1.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv2.weight = nn.Parameter(torch.rand(8, 8, 3, 3)-.5)
        self.conv2.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv3.weight = nn.Parameter(torch.rand(8,8,3,3)-.5)
        self.conv3.bias = nn.Parameter(torch.rand(8)-.5)

        self.lin.weight = nn.Parameter(torch.rand(2,128)-.5)
        self.lin.bias = nn.Parameter(torch.rand(1,2)-.5)
        self.smax = nn.Softmax()
    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """

        f1 = self.conv1(x)
        #print(f1.shape)
        f2 = self.pool1(self.relu(f1))
        #print(f2.shape)
        f3 = self.conv2(f2)
        #print(f3.shape)
        f4 = self.conv3(self.relu(f3))
        #print(f4.shape)
        f5 = self.lin(self.relu(f4).view(x.shape[0], 128))
        return f5

def load_puzzle(pgn_handle):
    """
    Intended use case seen in fit():
    @param pgn_handle: file handle for your training file
    """
    board = chess.Board()
    game = chess.pgn.read_game(pgn_handle)
    if game is None:
        return None, None
    fen = game.headers['FEN']
    board.set_fen(fen)
    move = None
    for j, mv in enumerate(game.mainline_moves()):
        if j == 0:
            board.push(mv)
        if j == 1:
            return board, mv

def fit():
    """
    This is just a snippet for reading board-move pairs you might use for training
    """
    with open('tactics.pgn') as pgn_handle:
        b, m = load_puzzle(pgn_handle)
        while b is not None:
            b, m = load_puzzle(pgn_handle)

boss = torch.load("./model.pb").cpu()
model = np.array([-148855.02,-30477.85,-30551.98,29841.76,4662.586,22795.406,16788.047,-390.46484,16790.209,20671.39,30467.434,-15475.637,18777.492,-7198.05,834.36914,7917.33,91261.9,39197.73,1681.3584,7309.202,21264.67,-48650.93,-16695.719,-34107.133,-32389.525,-27868.287,2821.3887,28119.285,-17444.44,19453.479,-12166.459,-33944.89,-50989.258,21926.771,37144.582,-774.3887,67195.1,-24727.703,3942.705,-17738.887,14451.726,-28555.973,38237.65,16417.725,1101.3926,2892.3496,2789.4268,-61240.38,-17554.371,22682.945,-32756.129,2747.2783,-40330.312,767.0371,-7597.7646,18127.416,-17100.932,13093.983,3943.0972,-17896.686,3307.8916,-14772.504,17312.781,-14401.801,-14675.602,63562.777,-4818.365,12779.569,-34392.31,4150.4473,-5685.792,-11512.883,12125.723,7598.699,-16430.812,9237.025,-1396.5015,14037.002,-3727.7148,3356.7769,57096.2,23020.168,61946.97,20271.76,39612.594,-21487.064,-14213.066,-13672.64,-281.10938,8981.791,14391.947,-15882.488,-22090.688,-5796.4746,-3400.6191,25483.62,-44322.016,-750.1787,-15342.653,-26312.316,-56273.04,9156.345,25591.912,-22664.91,-1460.2793,15694.838,6385.2344,-1822.6992,25831.07,21973.793,-15164.575,13941.137,25929.984,2994.1406,4631.8584,-25880.756,-26683.645,10185.639,-14738.164,10518.768,-35096.473,28373.113,-41187.953,-5838.953,-11273.999,-15915.155,7152.2856,23831.645])

def move(board):
    global boss
    global model
    lowest_op_score = 1500
    best_move = None
    for move in board.legal_moves:
        board.push(move)

        b = torch.Tensor(encode_board(board)).view(1,13,8,8)
        features = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(b)))))).view(b.shape[0], 128).detach().numpy().reshape(128)
        #features = np.concatenate([features, np.array([1])])
        move_score = sum(features * model) + 152955.34
        print(move)
        print(move_score)
        if move_score < lowest_op_score:

            lowest_op_score = move_score
            best_move = move
        board.pop()
    return best_move

memo = {}
skip_count = 0
def get_resulting_boards(board, d):
    global memo
    global skip_count
    boards = {}
    if d == 1:
        for move in board.legal_moves:
            board.push(move)
            boards[board.fen()]=1
            board.pop()
        return boards

    else:
        for move in board.legal_moves:
            temp = board.copy()
            temp.push(move)
            fen = temp.fen()
            if fen not in memo:
                memo[fen] = d
            else:
                if memo[fen] >=d:
                    skip_count+=1
                    continue
                else:
                    memo[fen] = d

            for r_board in get_resulting_boards(temp, d-1):
                boards[r_board]=1

        return boards

def score_board(board):
    #pieces controlled
    #pieces attacked
    #squares controlled, prioritize center and deep

    p_score = 0

    atk_score = 0
    def_score = 0
    ctrl_score = 0
    b = board

    p_vals = [1,3,3,5,9,50]

    def_vals = [1,1,1,1,1,0]

    ctrl_row = [10,12,16,25,25,16,12,10]
    ctrl_col = [.1,.1,.1,.2,.25,.25,.25,.25]

    ctrl_vals = []

    for i in range(8):
        for j in range(8):
            ctrl_vals.append(ctrl_row[j] * ctrl_col[i])

    for i in range(6):
        for square in b.pieces(i+1, True):
            p_score+=p_vals[i]
        for square in b.pieces(i+1, False):
            p_score-=p_vals[i]

    for i in range(64):
        p = b.piece_at(i)
        if p is None:
            continue
        t = p.piece_type
        if p.color:
            for square in b.attacks(i):
                p2 = b.piece_at(square)
                if p2 is not None:
                    if p2.color:
                        def_score += def_vals[p.piece_type-1]/p_vals[p2.piece_type-1]
                    else:
                        atk_score += min(0, p_vals[p2.piece_type-1] - p_vals[p.piece_type-1])
                ctrl_score += ctrl_vals[square]/p_vals[p.piece_type-1]
        else:
            for square in b.attacks(i):
                p2 = b.piece_at(square)
                if p2 is not None:
                    if p2.color:
                        atk_score -= min(0, p_vals[p2.piece_type-1] - p_vals[p.piece_type-1])
                    else:
                        def_score -= def_vals[p.piece_type-1]/p_vals[p2.piece_type-1]
                ctrl_score -= ctrl_vals[63-square]/p_vals[p.piece_type-1]
    return p_score, atk_score, def_score, ctrl_score, board.turn


def encode_board(b):
    x = np.zeros((13,8,8))

    for i in range(64):
        p = b.piece_at(i)
        if p is None:
            x[0][int(i/8)][i%8]=1
        else:
            if (p.color and b.turn) or (not p.color and not b.turn):
                x[p.piece_type][int(i/8)][i%8] = 1
            else:
                x[p.piece_type+6][int(i/8)][i%8] = 1

    return x

if __name__ == "__main__":
    b= chess.Board()
    print(move(b))
