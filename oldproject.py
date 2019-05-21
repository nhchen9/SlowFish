
import chess
import chess.pgn
import chess.engine
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.linear_model import LinearRegression


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

boss = torch.load("v093.h5").cpu()
model = pickle.load(open("chessnet93", 'rb'))

def move(board):
    global boss
    globel model
    lowest_op_score = 1500
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        b = torch.Tensor(project.encode_board(board)).view(1,13,8,8)
        features = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(x_test)))))).view(x_test.shape[0], 128).detach().numpy()

        move_score = model.predict(features)

        if move_score < lowest_op_score:
            lowest_op_score = move_score
            best_move = move
        board.pop()

b = chess.Board()

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

'''
print(len(get_resulting_boards(chess.Board(), 4)))
print(skip_count)
'''

if __name__ == "__main__":
    x_test = []
    y_test = []
    '''
    engine = chess.engine.SimpleEngine.popen_uci("./stockfish.exe")
    i=0
    with open('tactics.pgn') as pgn_handle:
        b, m = load_puzzle(pgn_handle)
        while b is not None:
            i+=1
            if i%100 ==0:
                print(i)

            b, m = load_puzzle(pgn_handle)
            if b is None:
                break
            x_test.append(encode_board(b))
            y_test.append(max(min(engine.analyse(b, chess.engine.Limit(depth=12))['score'].white().score(mate_score=1500), 1500), -1500))
            if not b.turn:
                y_test[len(y_test)-1] *= -1

    engine.quit()

    np.save("y_test.npy", y_test)
    print(y_test)
    '''
    x = np.load("x_test.npy")
    y = np.load("y_test.npy")
    x = torch.Tensor(x)
    #y = torch.Tensor([i > 0 for i in y])

    x_train = x[0:700,]
    x_test = x[700:(len(x)-1),]
    y_train = y[0:700]
    y_test = y[700:(len(y)-1)]
    #print(x_train.shape)
    boss =  torch.load("v5bin_chess_net5.h5").cpu()

    x_train_feat = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(x_train)))))).view(x_train.shape[0], 128).detach().numpy()
    x_test_feat = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(x_test)))))).view(x_test.shape[0], 128).detach().numpy()

    x_train_feat2 = boss2.conv3(boss2.relu(boss2.conv2(boss2.pool1(boss2.relu(boss2.conv1(x_train)))))).view(x_train.shape[0], 128).detach().numpy()
    x_test_feat2 = boss2.conv3(boss2.relu(boss2.conv2(boss2.pool1(boss2.relu(boss2.conv1(x_test)))))).view(x_test.shape[0], 128).detach().numpy()

    reg = LinearRegression().fit(x_train_feat, y_train)
    y1 = reg.predict(x_test_feat)

    reg = LinearRegression().fit(x_train_feat2, y_train)
    y2 = reg.predict(x_test_feat2)

    l1 = sum(abs(y1-y_test))/len(y_test)
    print(y_test)
    print(y1)
    print(y2)
    print(l1)

    l2 = sum(abs(y2-y_test))/len(y_test)
    print(l2)
    '''
    pred = myFish(x_test)
    print(pred[0:100,])
    print(y_test[0:100])
    labels = torch.Tensor([torch.max(row, 0)[1] for row in pred])
    print("val imbalance:", sum(labels)/len(labels))
    error = float(sum(abs(y_test.float()-labels)))
    total = y_test.shape[0]

    print("error:", error/total)
    '''

    '''
    for n in range(2,9):
        X_train = []
        y_train = np.load("y_12fen" + str(n) + ".npy")
        #Y_train = []
        fens = np.load("fen_list"+str(n)+".npy")
        #engine = chess.engine.SimpleEngine.popen_uci("./stockfish.exe")
        i=0
        for f in fens:

            if i %1000 ==0:
                print(i)
            b=chess.Board(fen = f)
            #Y_train.append(max(min(engine.analyse(b, chess.engine.Limit(depth=8))['score'].white().score(mate_score=2000), 2000), -2000))
            #X_train.append(encode_board(b))
            if not b.turn:
                y_train[i] *= -1
            i+=1

        #engine.quit()
        #X_train = np.array(X_train)
        y_train = np.array(y_train)

        #np.save("x_genfen12coded"+str(n)+".npy", X_train)
        np.save("y_genfen12flipped"+str(n)+".npy", y_train)
    '''
    '''

    class SlowFish(nn.Module):
        def __init__(self):
            super(SlowFish, self).__init__()
            d = 20
            self.lin1 = nn.Linear(5,d, bias = True)
            self.lin2 = nn.Linear(d,d, bias = True)
            self.lin3 = nn.Linear(d,1, bias = True)
            self.lin1.weight = nn.Parameter(torch.rand(d,5)-.5, requires_grad=True)
            self.lin2.weight = nn.Parameter(torch.rand(d,d)-.5, requires_grad=True)
            self.lin3.weight = nn.Parameter(torch.rand(1,d)-.5, requires_grad=True)
            self.lin1.bias = nn.Parameter(torch.rand((d))-.5, requires_grad=True)
            self.lin2.bias = nn.Parameter(torch.rand((d))-.5, requires_grad=True)
            self.lin3.bias = nn.Parameter(torch.rand((1))-.5, requires_grad=True)
            self.sig = nn.Sigmoid()
        def forward(self, xb):
            z = self.lin3(self.sig(self.lin2(self.sig(self.lin1(xb)))))
            return z


    def fit(net, optimizer,  X, Y, n_epochs):
        loss_fn = nn.MSELoss() #note: input to loss function needs to be of shape (N, 1) and (N, 1)
        with torch.no_grad():
            epoch_loss = [loss_fn(net(X), Y)]
        for e in range(n_epochs):
            pred = net(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if e %100==0:
                print(loss)


            epoch_loss.append(loss)
            #TODO: compute the loss for X, Y, it's gradient, and optimize
            #TODO: append the current loss to epoch_loss
        return epoch_loss
    torch.cuda.set_device(0)


    loss_fn = nn.MSELoss()
    train_X = torch.from_numpy(np.load("x_traind12full.npy")).float().view(9574, 5).cuda()
    train_Y = torch.from_numpy(np.load("y_traind12full.npy")).float().view(9574, 1).cuda()

    myFish = SlowFish().cuda()
    print(train_X)
    print(train_Y)
    optimizer = optim.SGD(myFish.parameters(), lr = 0.005, momentum=0.1)
    fit(myFish, optimizer, train_X, train_Y, 3000)
    '''
