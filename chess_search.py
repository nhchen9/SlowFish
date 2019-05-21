import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import project
import pickle
from sklearn.preprocessing import OneHotEncoder


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

boss = torch.load("chessnet2.0/v093.h5").cpu()
model = pickle.load(open("chessnet93", 'rb'))
def move_search(board):

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
