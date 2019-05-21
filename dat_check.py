import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

y2=torch.Tensor(np.load("y_genfen2.npy"))
y4=torch.Tensor(np.load("y_genfen4.npy"))
y8=torch.Tensor(np.load("y_genfen8.npy"))
y12=torch.Tensor(np.load("y_genfen12.npy"))
y16=torch.Tensor(np.load("y_genfen16.npy"))
y20=torch.Tensor(np.load("y_genfen20.npy"))

z8=torch.Tensor(np.load("y_genfen8p2.npy"))
z12=torch.Tensor(np.load("y_genfen12p2.npy"))
z16=torch.Tensor(np.load("y_genfen16p2.npy"))

plt.hist(y8)
plt.show()

plt.hist(z8)
plt.show()

plt.hist(y12)
plt.show()

plt.hist(z12)
plt.show()

plt.hist(y16)
plt.show()


plt.hist(z16)
plt.show()

loss = nn.MSELoss()

print(loss(y2, y20))
print(loss(y4, y20))
print(loss(y8, y20))
print(loss(y12, y20))
print(loss(y16, y20))
