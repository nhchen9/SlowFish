import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, c_in, c_out):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1, bias = False)
        self.batch1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias = False)
        self.batch2 = nn.BatchNorm2d(c_out)

        self.conv1.weight = nn.Parameter(torch.rand(self.conv1.weight.shape))
        self.conv2.weight = nn.Parameter(torch.rand(self.conv2.weight.shape))
        self.batch1.weight = nn.Parameter(torch.rand(self.batch1.weight.shape))
        self.batch1.bias = nn.Parameter(torch.rand(self.batch1.bias.shape))
        self.batch2.weight = nn.Parameter(torch.rand(self.batch2.weight.shape))
        self.batch2.bias = nn.Parameter(torch.rand(self.batch2.bias.shape))

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """

        return(self.relu(x + self.batch2(self.conv2(self.relu(self.batch1(self.conv1(x)))))))

    def set_param(self, kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_1: a (C, C, 3, 3) tensor, kernels of the first conv layer.
            bn1_weight: a (C,) tensor.
            bn1_bias: a (C,) tensor.
            kernel_2: a (C, C, 3, 3) tensor, kernels of the second conv layer.
            bn2_weight: a (C,) tensor.
            bn2_bias: a (C,) tensor.
        """


        self.conv1.weight = nn.Parameter(kernel_1)
        self.conv2.weight = nn.Parameter(kernel_2)
        self.batch1.weight = nn.Parameter(bn1_weight)
        self.batch1.bias = nn.Parameter(bn1_bias)
        self.batch2.weight = nn.Parameter(bn2_weight)
        self.batch2.bias = nn.Parameter(bn2_bias)


class chessNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self):

        super(chessNet, self).__init__()
        self.conv1 = nn.Conv2d(13, 8, 3, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1, bias = True)
        self.conv3 = nn.Conv2d(8, 8, 3, stride = 1, padding = 1, bias = True)
        self.lin = nn.Linear(128, 1)

        print(self.conv1.weight.shape)
        print(self.conv1.bias.shape)

        self.conv1.weight = nn.Parameter(torch.rand(8, 13, 3,3)-.5)
        self.conv1.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv2.weight = nn.Parameter(torch.rand(8, 8, 3, 3)-.5)
        self.conv2.bias = nn.Parameter(torch.rand(8)-.5)
        self.conv3.weight = nn.Parameter(torch.rand(8,8,3,3)-.5)
        self.conv3.bias = nn.Parameter(torch.rand(8)-.5)

        self.lin.weight = nn.Parameter(torch.rand(1,128)-.5)
        self.lin.bias = nn.Parameter(torch.rand(1,1)-.5)

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
        f5 = self.lin(f4.view(x.shape[0], 128))
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
        if i%10 ==0:
            torch.save(net, "bin_chess_net"+str(i)+".h5")
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
            for X, Y in val_dl:
                print(net(X))
                break
            for X,Y in train_dl:
                pred = net(X)
                labels = torch.Tensor([torch.max(row, 0)[1] for row in pred]).cuda()
                error += float(sum(abs(Y.float()-labels)))
                total += Y.shape[0]
                del labels
                break
            print("train error:", error/total)
            print("bias:", sum(pred))
            error=0
            total=0

            for X,Y in val_dl:
                pred = net(X)
                labels = torch.Tensor([torch.max(row, 0)[1] for row in pred]).cuda()
                error += float(sum(abs(Y.float()-labels)))
                total += Y.shape[0]
                del labels
                break
            print("val error:", error/total)
            print("bias:", sum(pred))
            print("train loss:", train_epoch_loss[i+1])
            print("val loss:", val_epoch_loss[i+1])
    return train_epoch_loss, val_epoch_loss
if __name__ == "__main__":
    n=100000
    myFish = torch.load("bin_chess_net20.h5").cuda()
    for name, param in myFish.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    x_train = torch.Tensor(np.load("x_genfen12coded2.npy")[0:n,]).cuda()
    y_train = torch.Tensor(np.load("y_genfen12flipped2.npy")[0:n,])
    #y_train = y_train /1500
    y_train = torch.Tensor([z > 0 for z in y_train]).cuda().long()
    train = torch.utils.data.TensorDataset(x_train,y_train)

    x_val = torch.Tensor(np.load("x_genfen12coded3.npy")[0:n,]).cuda()
    y_val = torch.Tensor(np.load("y_genfen12flipped3.npy")[0:n,])
    #y_val = y_val /1500
    y_val = torch.Tensor([z>0 for z in y_val]).cuda().long()

    val = torch.utils.data.TensorDataset(x_val,y_val)

    print(myFish(x_train))
    optimizer = optim.SGD(myFish.parameters(), lr = 0.0002, momentum=0.8)


    tl, vl = fit_and_validate(myFish, F.cross_entropy, optimizer, train, val, 1000, 10000)
