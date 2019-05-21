
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

class moveNet(nn.Module):

    def __init__(self):

        super(moveNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, [3,3], stride = 1, padding = 1, bias = False)
        #self.batch1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d([13,1])
        self.conv2 = nn.Conv2d(4, 4, [3,3], stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(4, 4, 3, stride = 1, padding = 0, bias = True)

        self.pool2 = nn.MaxPool2d(2)
        self.lin = nn.Linear(36, 64**2)
        self.soft =nn.Softmax(dim=1)
        '''
        self.conv1.weight = nn.Parameter(torch.rand((4,1,3,3))-.5)
        self.batch1.weight = nn.Parameter(torch.rand((4,))-.5)
        self.batch1.bias = nn.Parameter(torch.rand((4,))-.5)

        self.conv2.weight = nn.Parameter(torch.rand((4,4,3,3))-.5)
        self.conv2.bias = nn.Parameter(torch.rand(4,)-.5)

        self.conv3.weight = nn.Parameter(torch.rand((4,4,3,3))-.5)
        self.conv3.bias = nn.Parameter(torch.rand(4,)-.5)

        self.lin.weight = nn.Parameter(torch.rand((36,64*64))-.5)
        '''
        '''
        for param in self.parameters():
            param.data = param.data.half()
        '''
    def forward(self, x):
        #print(x.element_size() * x.nelement())
        f1 = self.conv1(x.view((x.shape[0], 1, 104, 8)))
        f2 = self.pool1(self.relu(f1))
        f3 = self.conv2(f2)
        f4 = self.conv3(self.relu(f3))
        f5 = self.pool2(f4)

        f6 = self.lin(f5.view((x.shape[0], 36)))
        #f7 = self.soft(f6)
        return f6

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

boss = moveNet().cpu()
boss.load_state_dict(torch.load('./model.pb'))

#model = np.array([-7769.5903,4765.5957,-5004.4736,1026.6111,4659.403,3823.5654,-2487.5137,-1819.5117,-2662.2812,13146.136,4908.0547,23881.908,-21569.52,1258.228,-13191.488,-19038.889,710.646,-3358.4863,-11264.898,-10882.581,10239.848,12429.692,-4613.2935,9210.758,-15207.444,-3708.3665,9170.324,-8426.023,4283.7466,418.66357,-13192.726,-17978.994,-7425.7935,-28591.54,-2582.3643,-3501.7412,2647.6538,24239.893,13401.092,14924.878,-783.3496,-3473.7383,-4941.208,12075.254,-6386.3936,7746.6924,-909.3983,-28629.305,1469.6279,-4335.9707,23235.11,29608.664,-16164.692,3264.8206,-10675.063,2977.4668,4303.235,-4217.3657,10689.739,-17396.016,-13641.766,-17114.654,-5900.4727,5736.9023,-9558.045,-3233.042,3528.0747,14223.741,5961.5205,-642.1655,2495.8682,-12721.611,-3549.969,-9843.097,-2660.9036,6087.6763,2175.0986,-7266.8833,2361.5635,9628.732,-5532.1064,-2364.788,457.7085,-3432.5645,21456.832,19692.902,-4273.537,-20131.93,-11475.2705,-11063.887,6724.668,25647.395,1217.2329,-3655.8562,-2581.0464,-23271.143,-6286.9805,12271.127,-15481.478,-10482.597,8211.805,-3864.8179,-3339.563,13948.095,-5216.4707,7455.1074,-413.8711,-7934.5903,1913.7222,3441.7344,12771.908,-18237.184,-5231.0566,-6208.752,-3072.2627,24453.027,384.65234,-3110.0098,-1965.6294,-6586.7427,-12661.216,15975.135,-9790.668,-3613.373,-10082.783,5628.8135,17368.215,3349.3486])
#intercept = 6581.0713
def move(board):
    global boss
    x = torch.Tensor(encode_board(board)).view((1,104,8))
    pred = boss(x).view((4096))
    i = torch.max(pred, 0)[1]
    return decode_move(int(i))

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


def old_encode_board(b):
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

def encode_board(b):
    x = np.zeros((13*8,8), dtype = np.uint8)

    for i in range(64):
        p = b.piece_at(i)
        if p is None:
            x[13*int(i/8)][i%8]=1
        else:
            if (p.color and b.turn) or (not p.color and not b.turn):
                x[p.piece_type+ 13*int(i/8)][i%8] = 1
            else:
                x[p.piece_type+6+13*int(i/8)][i%8] = 1
    if not b.turn:
        b2 = chess.Board()
        for i in range(64):
            b2.set_piece_at(i, b.piece_at(63-i))
    return x

def num_to_piece(n):
    if n==0:
        return None
    else:
        if n < 7:
            return chess.Piece(n, True)
        else:
            return chess.Piece(n-6, False)

def decode_board(x):
    b = chess.Board()
    for i in range(64):
        start = 13*int(i/8)
        slice = list(x[start:start+13,i%8])
        b.set_piece_at(i, num_to_piece(slice.index(1)))
    return b

def decode_move(y):

    e = y%64
    s = int(y/64)
    start = [int(s/8), s%8]

    end = [int(e/8), e%8]

    return(chr(start[0]+97)+ str(start[1]+1) + chr(end[0]+97) + str(end[1]+1))

if __name__ == "__main__":
    z = np.load("datasets/x_train_moves2.npy")
    moves = np.load("datasets/y_train_moves2.npy").astype(int)
    x = np.array(z, dtype = np.uint8)
    del z
    right =0
    for i in range(100000):
        b = decode_board(x[i])
        #print(b)
        #print(move(b))
        #print(decode_move(moves[i]))
        if move(b) == decode_move(moves[i]):
            right+=1
    print(right/100000)


    '''
    print(decode_board(x[1]))
    moves = np.load("datasets/y_train_moves2.npy").astype(int)
    print(moves.shape)
    train_ind = int(len(moves)*.75)
    b = chess.Board()
    print(move(b))
    '''
