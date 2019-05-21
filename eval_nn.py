import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

class moveNet(nn.Module):

    def __init__(self):

        super(moveNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, [3,3], stride = 1, padding = 1, bias = False)
        self.batch1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d([13,1])
        self.conv2 = nn.Conv2d(4, 4, [3,3], stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(4, 4, 3, stride = 1, padding = 0, bias = True)
        self.pool2 = nn.MaxPool2d([2,2])
        self.lin = nn.Linear(48, 64**2)

    def forward(self, x):
        f1 = self.batch1(self.conv1(x))

        f2 = self.pool1(self.relu(f1))

        f3 = self.conv2(f2)

        f4 = self.conv3(self.relu(f3))

        f5 = self.pool2(f4)

        f6 = self.lin(f5)

        return f6

class ChessNet(nn.Module):

    def __init__(self):

        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 8, 3, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1, bias = True)
        self.lin = nn.Linear(128, 1)
        '''
        boss_net = torch.load("chessnet2.0/v093.h5")

        self.conv1.weight = boss_net.conv1.weight
        self.conv1.bias = boss_net.conv1.bias
        self.conv2.weight = boss_net.conv2.weight
        self.conv2.bias = boss_net.conv2.bias
        self.conv3.weight = boss_net.conv3.weight
        self.conv3.bias = boss_net.conv3.bias

        for param in self.parameters():
            param.requires_grad = False
        '''
        self.conv1.weight = nn.Parameter(torch.rand(8, 13, 3,3)-.5)
        self.conv1.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv2.weight = nn.Parameter(torch.rand(8, 8, 3, 3)-.5)
        self.conv2.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv3.weight = nn.Parameter(torch.rand(8,8,3,3)-.5)
        self.conv3.bias = nn.Parameter(torch.rand(8)-.5)

        self.lin.weight = nn.Parameter(torch.rand(1,128)-.5, requires_grad = True)
        self.lin.bias = nn.Parameter(torch.rand(1,1)-.5, requires_grad = True)
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


        #return self.lin(self.adapt(self.block((self.relu(self.batch1(self.conv1(x)))))).view(x.shape[0], 6))


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


class binaryChessNetv2(nn.Module):
    """A simplified ResNet."""

    def __init__(self):

        super(binaryChessNetv2, self).__init__()
        self.conv1 = nn.Conv2d(13, 8, 3, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 6, 3, stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(6, 6, 4, stride = 1, padding = 1, bias = True)
        self.conv4 = nn.Conv2d(6, 4, 3, stride = 1, padding = 0, bias = True)
        self.lin = nn.Linear(16, 2)

        print(self.conv1.weight.shape)
        print(self.conv1.bias.shape)

        self.conv1.weight = nn.Parameter(torch.rand(8, 13, 3,3)-.5)
        self.conv1.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv2.weight = nn.Parameter(torch.rand(6, 8, 3, 3)-.5)
        self.conv2.bias = nn.Parameter(torch.rand(6)-.5)
        self.conv3.weight = nn.Parameter(torch.rand(6,6,3,3)-.5)
        self.conv3.bias = nn.Parameter(torch.rand(6)-.5)
        self.conv4.weight = nn.Parameter(torch.rand(4,6,3,3)-.5)
        self.conv4.bias = nn.Parameter(torch.rand(4)-.5)



        self.lin.weight = nn.Parameter(torch.rand(2,16)-.5)
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
        f5 = self.conv4(self.relu(f4))
        #print(f5.shape)
        f6 = self.lin(self.relu(f5).view(x.shape[0], 16))
        #print(f6.shape)
        return f6

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 6, 3, stride = 1, padding = 1, bias = False)
        self.batch1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.block = Block(6, 6)
        self.conv2 = nn.Conv2d(6, 3, 3, stride = 1, padding = 0, bias = True)
        self.adapt = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Linear(3, 1)

        self.conv1.weight = nn.Parameter(torch.rand(self.conv1.weight.shape))
        self.conv2.weight = nn.Parameter(torch.rand(self.conv2.weight.shape))
        self.conv2.bias = nn.Parameter(torch.rand(self.conv2.bias.shape))
        self.batch1.weight = nn.Parameter(torch.rand(self.batch1.weight.shape))
        self.batch1.bias = nn.Parameter(torch.rand(self.batch1.bias.shape))
        self.lin.weight = nn.Parameter(torch.rand(self.lin.weight.shape))
        self.lin.bias = nn.Parameter(torch.rand(self.lin.bias.shape))

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """

        f1 = self.conv1(x)
        #print(f1.shape)
        f2 = self.batch1(f1)
        #print(f2.shape)
        f3 = self.pool1(self.relu(f2))
        #print(f3.shape)
        f4 = self.block(f3)
        #print(f4.shape)
        #f5 = self.pool2(f4)
        #print(f5.shape)
        f6 = self.conv2(f4)
        #print(f6.shape)
        f7 = self.adapt(self.relu(f6))
        #print(f7.shape)
        f8 = self.lin(f7.view(x.shape[0], 3))


        return f8

        return self.lin(self.adapt(self.block((self.relu(self.batch1(self.conv1(x)))))).view(x.shape[0], 6))

    def set_param(self, kernel_0, bn0_weight, bn0_bias,
                  kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias,
                  fc_weight, fc_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_0: a (C, 1, 3, 3) tensor, kernels of the conv layer
                      before the building block.
            bn0_weight: a (C,) tensor, weight of the batch norm layer
                        before the building block.
            bn0_bias: a (C,) tensor, bias of the batch norm layer
                      before the building block.
            fc_weight: a (10, C) tensor
            fc_bias: a (10,) tensor
        See the docstring of Block.set_param() for the description
        of other arguments.
        """
        print('hi')

def loss_batch(model, loss_func, xb, yb, opt=None):
    """ Compute the loss of the model on a batch of data, or do a step of optimization.

    @param model: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer.  If not None, use the Optimizer to improve the model. Otherwise, just compute the loss.
    @return a numpy array of the loss of the minibatch, and the length of the minibatch
    """
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
            print(net(X))
            print(Y)
            break
        t_loss = torch.Tensor([loss_func(net(X),Y) for X, Y in train_dl]).mean()
        train_epoch_loss = [t_loss]
        v_loss = torch.Tensor([loss_func(net(X),Y) for X, Y in val_dl]).mean()
        val_epoch_loss = [v_loss]
        print(train_epoch_loss)
        print(val_epoch_loss)
        # TODO compute the validation loss and store it in a list
    for i in range(n_epochs):
        print("base epoch #", i)
        if i%3 ==0:
            torch.save(net, "chessnet2.0/v0"+str(i)+".h5")
        net.train() #put the net in train mode
        for X,Y in train_dl:
            pred = net(X)
            #print(pred)
            loss = loss_func(pred, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            train_epoch_loss.append(torch.Tensor([loss_func(net(X),Y) for X, Y in train_dl]).mean())
            val_epoch_loss.append(torch.Tensor([loss_func(net(X),Y) for X, Y in val_dl]).mean())
            #print(train_epoch_loss[i])
            #print(val_epoch_loss[i])
            error=0
            total=0
            for X,Y in train_dl:
                pred = net(X)
                labels = torch.Tensor([torch.max(row, 0)[1] for row in pred]).cuda()
                print("train imbalance:", sum(labels)/len(labels))
                error += float(sum(abs(Y.float()-labels)))
                total += Y.shape[0]
                del labels
                break
            print("train error:", error/total)
            #print("bias:", sum(pred))
            error=0
            total=0

            for X,Y in val_dl:
                pred = net(X)
                labels = torch.Tensor([torch.max(row, 0)[1] for row in pred]).cuda()
                print("val imbalance:", sum(labels)/len(labels))
                error += float(sum(abs(Y.float()-labels)))
                total += Y.shape[0]
                del labels
                break
            print("val error:", error/total)
            #print("bias:", sum(pred))
            print("train loss:", train_epoch_loss[i+1])
            print("val loss:", val_epoch_loss[i+1])

    return train_epoch_loss, val_epoch_loss


def reg_fit_and_validate(net, loss_func, optimizer, train, val, n_epochs, batch_size =100):
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
            print(net(X))
            print(Y)
            break

        t_loss = torch.Tensor([loss_func(net(X),Y) for X, Y in train_dl]).mean()
        train_epoch_loss = [t_loss]
        v_loss = torch.Tensor([loss_func(net(X),Y) for X, Y in val_dl]).mean()
        val_epoch_loss = [v_loss]
        print(train_epoch_loss)
        print(val_epoch_loss)
        # TODO compute the validation loss and store it in a list
    for i in range(n_epochs):
        print("base epoch #", i)
        if i%3 ==0:
            torch.save(net, "chessregnet/v0"+str(i)+".h5")
        net.train() #put the net in train mode
        for X,Y in train_dl:
            pred = net(X)
            #print(pred)
            loss = loss_func(pred, Y)
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        #print (pred[0:100,])
        #print (Y[0:100])
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            train_epoch_loss.append(torch.Tensor([loss_func(net(X),Y) for X, Y in train_dl]).mean())
            val_epoch_loss.append(torch.Tensor([loss_func(net(X),Y) for X, Y in val_dl]).mean())
            #print(train_epoch_loss[i])
            #print(val_epoch_loss[i])

            print("train loss:", train_epoch_loss[i+1])
            print("val loss:", val_epoch_loss[i+1])

if __name__ == "__main__":

    n= 10000
    x_train = torch.cat((torch.Tensor(np.load("x_genfen12coded2.npy")[0:n,]), torch.Tensor(np.load("x_genfen12coded3.npy")[0:n,]), torch.Tensor(np.load("x_genfen12coded4.npy")[0:n,])))
    y_train = torch.cat((torch.Tensor(np.load("y_genfen12flipped2.npy")[0:n,]), torch.Tensor(np.load("y_genfen12flipped3.npy")[0:n,]), torch.Tensor(np.load("y_genfen12flipped4.npy")[0:n,])))


    #x_train = torch.Tensor(np.load("x_genfen12coded2.npy")[0:n,]).cuda()
    #y_train = torch.Tensor(np.load("y_genfen12flipped2.npy")[0:n,])

    #y_train = torch.Tensor([z > 0 for z in y_train]).cuda().long()


    x_val = torch.Tensor(np.load("x_genfen12coded6.npy")[0:n,])
    y_val = torch.Tensor(np.load("y_genfen12flipped6.npy")[0:n,])

    #y_val = torch.Tensor([z>0 for z in y_val]).cuda().long()

    train = torch.utils.data.TensorDataset(x_train,y_train)
    val = torch.utils.data.TensorDataset(x_val,y_val)

    boss = torch.load("chessnet2.0/v093.h5").cpu()
    boss2 =  torch.load("v5bin_chess_net50.h5").cpu()

    x_train_feat = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(x_train)))))).view(x_train.shape[0], 128).detach().numpy()
    x_val_feat = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(x_val)))))).view(x_val.shape[0], 128).detach().numpy()

    x_train_feat2 = boss2.conv3(boss2.relu(boss2.conv2(boss2.pool1(boss2.relu(boss2.conv1(x_train)))))).view(x_train.shape[0], 128).detach().numpy()
    x_val_feat2 = boss2.conv3(boss2.relu(boss2.conv2(boss2.pool1(boss2.relu(boss2.conv1(x_val)))))).view(x_val.shape[0], 128).detach().numpy()
    #myFish = torch.load("v4bin_chess_net14.h5").cuda()
    '''
    train_feat = torch.utils.data.TensorDataset(x_train_feat,y_train)
    val_feat = torch.utils.data.TensorDataset(x_val_feat,y_val)

    lin = nn.Linear(128,1)
    lin.weight = nn.Parameter(torch.rand(1,128)-.5)
    lin.bias = nn.Parameter(torch.rand(1,1 )-.5)

    lin.cuda()

    myFish = ChessNet().cuda()

    #print(myFish(x_train))
    optimizer = optim.SGD(lin.parameters(), lr = .1, momentum=0.9)
    '''
    y_train = y_train.cpu().detach().numpy()
    y_val = y_val.cpu().detach().numpy()

    reg1 = LinearRegression().fit(x_train_feat, y_train)
    y1 = reg1.predict(x_val_feat)

    l1 = sum(abs(y1-y_val))/len(y_val)
    print(l1)

    reg2 = LinearRegression().fit(x_train_feat2, y_train)
    y2 = reg2.predict(x_val_feat2)

    l2 = sum(abs(y2-y_val))/len(y_val)
    print(l2)

    x_test = torch.Tensor(np.load("x_test.npy"))
    y_test = torch.Tensor(np.load("y_test.npy"))


    x_test_feat = boss.conv3(boss.relu(boss.conv2(boss.pool1(boss.relu(boss.conv1(x_test)))))).view(x_test.shape[0], 128).detach().numpy()

    x_test_feat2 = boss2.conv3(boss2.relu(boss2.conv2(boss2.pool1(boss2.relu(boss2.conv1(x_test)))))).view(x_test.shape[0], 128).detach().numpy()


    yt1 = reg1.predict(x_test_feat)
    yt2 = reg2.predict(x_test_feat2)

    z1 = [i>0 for i in yt1]
    z2 = [i>0 for i in yt2]
    print(sum(yt1))
    print(sum(yt2))
    print(sum(z1))
    print(sum(z2))

    pickle.dump(reg1, open('chessnet93', 'wb'))
    pickle.dump(reg2, open('v5bin50', 'wb'))

    #tl, vl = reg_fit_and_validate(lin, nn.L1Loss(), optimizer, train_feat, val_feat, 1000, 10000)
