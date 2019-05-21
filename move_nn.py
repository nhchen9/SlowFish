import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
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
        self.conv4 = nn.Conv2d(4, 4, 3, stride = 1, padding = 0, bias = True)
        self.pool2 = nn.MaxPool2d(2)
        self.lin = nn.Linear(16, 64**2)
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
        f5 = self.conv4(self.relu(f4))
        f6 = self.pool2(f5)

        f7 = self.lin(f6.view((x.shape[0], 16)))
        #f7 = self.soft(f6)
        return f7


def loss_batch(model, loss_func, xb, yb, opt=None):

    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit_and_validate(net, loss_func, optimizer, train, val, n_epochs, batch_size =100):
    """
    @param net: the neural network
    @param optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
    @param train: a torch.utils.data.Dataset
    @param val: a torch.utils.data.Dataset
    @param n_epochs: the number of epochs over which to do gradient descent
    @param batch_size: the number of samples to use in each batch of gradient descent
    @return train_epoch_loss, validation_epoch_loss: two arrays of length n_epochs+1, containing the mean loss at the beginning of training and after each epoch
    """
    net.eval() #put the net in evaluation mode
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    val_dl = torch.utils.data.DataLoader(val, batch_size)
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        for X,Y in train_dl:
            X_temp = X.float().cuda()
            #print(net(X_temp))
            #print(Y)
            break
        train_epoch_loss = []
        val_epoch_loss = []
        #print(train_epoch_loss)
        #print(val_epoch_loss)
        # TODO compute the validation loss and store it in a list
    for i in range(n_epochs):
        print(torch.cuda.memory_allocated(0))
        print("base epoch #", i)
        if i%3 ==0:
            torch.save(net.state_dict(), "movenet/v2"+str(i)+".sd" )
        net.train() #put the net in train mode
        first = True
        for X,Y in train_dl:


            X_temp = X.float().cuda()
            pred = net(X_temp)
            #print(pred)
            Y_temp = Y.long().cuda()
            loss = loss_func(pred, Y_temp)
            loss.backward()
            if first:
                print(torch.norm(net.conv1.weight.grad,2))
                print(torch.norm(net.conv2.weight.grad,2))
                print(torch.norm(net.conv3.weight.grad,2))
                print(torch.norm(net.conv4.weight.grad,2))
                print(torch.norm(net.lin.weight.grad,2))
                first = False

            optimizer.step()
            optimizer.zero_grad()
            del Y_temp
            del X_temp
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            t_l = 0

            v_l = 0
            #print(train_epoch_loss[i])
            #print(val_epoch_loss[i])
            right=0
            total=0
            b = 0
            for X,Y in train_dl:
                b+=1
                X_temp = X.float().cuda()
                Y_temp = Y.long().cuda()
                pred = net(X_temp)
                t_l += loss_func(pred, Y_temp)
                labels = torch.Tensor([torch.max(row, 0)[1] for row in pred])
                for i in range(len(labels)):
                    if int(labels[i]) == int(Y[i].cpu()):
                        right+=1
                total += Y.shape[0]


                del labels
                del X_temp
                del Y_temp

                break
            t_l /= b
            print("train acc:", right/total)
            #print("bias:", sum(pred))
            right=0
            total=0
            b=0
            for X,Y in val_dl:
                b+=1
                X_temp = X.float().cuda()
                Y_temp = Y.long().cuda()
                pred = net(X_temp)
                v_l += loss_func(pred, Y_temp)
                labels = torch.Tensor([torch.max(row, 0)[1] for row in pred])
                for i in range(len(labels)):
                    if int(labels[i]) == int(Y[i].cpu()):
                        right+=1
                total += Y.shape[0]
                del labels
                del X_temp
                del Y_temp
                if i%10 == 0:
                    continue
                break
            v_l /= b

            train_epoch_loss.append(t_l)
            val_epoch_loss.append(v_l)
            if i%10 ==0:
                print("m10 val acc calc")
            print("val acc:", right/total)
            #print("bias:", sum(pred))

            print("train loss:", train_epoch_loss[len(train_epoch_loss)-1])
            print("val loss:", val_epoch_loss[len(train_epoch_loss)-1])

    return train_epoch_loss, val_epoch_loss


def byte_z1_loss(input, target):
    #print(input)
    #print(target)
    sub = torch.sub(input, target)
    #   print(torch.sum(torch.abs(sub)))
    return torch.sum(torch.abs(sub))

if __name__ == "__main__":
    z = np.load("datasets/flx_train_moves12.npy")
    x = np.array(z, dtype = np.uint8)
    del z

    moves = np.load("datasets/fly_train_moves12.npy").astype(int)
    print(moves.shape)
    train_ind = int(len(moves)*.75)

    '''
    y = np.zeros((len(moves),4096), dtype = np.uint8)

    for i in range(len(moves)):
        y[i][moves[i]] = 1


    '''
    '''
    y_train_temp = torch.Tensor(y[0:train_ind,]).short()
    ind = torch.nonzero(y_train_temp).t()
    vals = torch.zeros((ind.shape[1])) + 1

    y_train = torch.sparse.ByteTensor(ind, vals, y_train_temp.size()).cuda()

    y_val_temp = torch.Tensor(y[train_ind:len(moves),]).short()
    ind = torch.nonzero(y_val_temp).t()
    vals = torch.zeros((ind.shape[1])) + 1

    y_val = torch.sparse.ByteTensor(ind, vals, y_val_temp.size()).cuda()
    '''

    y_train = torch.ByteTensor(moves[0:train_ind,]).cuda()
    y_val = torch.ByteTensor(moves[train_ind:len(moves),]).cuda()
    #print(y_train.element_size())

    x_train = torch.ByteTensor(x[0:train_ind,]).cuda()
    x_val = torch.ByteTensor(x[train_ind:len(moves),]).cuda()
    #print(y_train.element_size()*y_train.nelement())
    #print(x_train.element_size()*x_train.nelement())
    del x
    '''
    x_train = torch.Tensor(x[0:train_ind,]).short().cuda()
    x_val = torch.Tensor(x[train_ind:len(moves),]).short().cuda()
    del x
    y_train = torch.Tensor(y[0:train_ind,]).short().cuda()
    y_val = torch.Tensor(y[train_ind:len(moves),]).short().cuda()
    '''
    print("data allocated")

    train = torch.utils.data.TensorDataset(x_train,y_train)
    val = torch.utils.data.TensorDataset(x_val,y_val)

    #myFish = torch.load("movenet/v051.h5")
    myFish = moveNet().cuda()
    myFish.load_state_dict(torch.load("movenet/v127.sd"))

    optimizer = optim.SGD(myFish.parameters(), lr = .05, momentum=0.9)
    print(torch.cuda.memory_allocated(0))
    tl, vl = fit_and_validate(myFish, F.cross_entropy, optimizer, train, val, 1000, 10000)
