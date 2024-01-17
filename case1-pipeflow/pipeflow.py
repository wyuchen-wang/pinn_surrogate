import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import random
import math
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import MSELoss
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# fluid parameters
rho = 1.
pi_tensor = torch.tensor(math.pi)

#nmae error
def nmae(y_pred, y_true):
  mae = torch.mean(torch.abs(y_true - y_pred))
  normalization_factor = torch.max(y_true) - torch.min(y_true)
  return mae / normalization_factor

def seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class XyDataset(Dataset):
    def __init__(self, X, y):
      self.X, self.y = X, y

    def __len__(self):
      return len(self.X)

    def __getitem__(self, idx):
      return self.X[idx], self.y[idx]

def load_data():
    data = {}
    for fn in os.listdir("miu_pipe"):
        miu = float(fn[3:fn.index("n")-1])
        data[miu] = np.load('miu_pipe/' + fn)
    return data

def train_test_split(data, miu_threshold= 1.3e-4, totensor=True):
    X_tr, y_tr, X_ts, y_ts = [], [], [], []

    for miu in data:
        x = data[miu]['x']
        y = data[miu]['y']
        u = data[miu]['u']
        v = data[miu]['v']
        p = data[miu]['p']
        N = len(x)

        input = np.concatenate([x, y, miu * np.ones((N, 1))], axis=1)
        output = np.concatenate([p, u, v], axis=1)

        if miu > miu_threshold:
            X_tr.append(input)
            y_tr.append(output)
        else:
            X_ts.append(input)
            y_ts.append(output)

    X_tr = np.concatenate(X_tr, axis=0)
    y_tr = np.concatenate(y_tr, axis=0)
    X_ts = np.concatenate(X_ts, axis=0)
    y_ts = np.concatenate(y_ts, axis=0)

    if totensor:
        return torch.FloatTensor(X_tr), torch.FloatTensor(X_ts), torch.FloatTensor(y_tr), torch.FloatTensor(y_ts)
    else:
        return X_tr, X_ts, y_tr, y_ts

def loss_data(X_batch, y_batch):
    x = X_batch[:, 0:1]
    y = X_batch[:, 1:2]
    miu = X_batch[:, 2:3]

    input = torch.cat((x, y, miu), 1)
    output = net(input)
    p = output[:, 0:1]
    u = output[:, 1:2]
    v = output[:, 2:3]

    # exact boundary
    # p_hard = p
    # u_hard = u
    # v_hard = v
    p_hard = 0.1 * (1 - x) - x * (1 - x) * p
    u_hard = u * (1 - y ** 2 / 0.05 ** 2)
    v_hard = v

    puv_pred = torch.cat((p_hard,u_hard,v_hard),1)
    loss = mse(puv_pred, y_batch)

    return loss

def loss_eqn(X_batch):
    x = X_batch[:,0:1]
    y = X_batch[:,1:2]
    miu = X_batch[:,2:3]

    x.requires_grad = True
    y.requires_grad = True
    miu.requires_grad = True

    input = torch.cat((x, y, miu), 1)
    output = net(input)
    p = output[:,0:1]
    u = output[:,1:2]
    v = output[:,2:3]

    #exact boundary
    # p_hard = p
    # u_hard = u
    # v_hard = v
    p_hard = 0.1 * (1 - x) - x * (1 - x) * p
    u_hard = u * (1 - y ** 2 / 0.05 ** 2)
    v_hard = v

    u_x = torch.autograd.grad(u_hard, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u_hard, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v_hard, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v_hard, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    p_x = torch.autograd.grad(p_hard, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    p_y = torch.autograd.grad(p_hard, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    em = u_x + v_y
    eu = u_hard*u_x + v_hard*u_y + 1/rho*p_x-miu/rho*(u_xx + u_yy)
    ev = u_hard*v_x + v_hard*v_y + 1/rho*p_y-miu/rho*(v_xx + v_yy)
    # em = u_x
    # eu = u_yy + 1 / (10 * miu)

    # loss
    loss_f = nn.MSELoss()
    loss = loss_f(em, torch.zeros_like(em)) \
           + loss_f(eu, torch.zeros_like(eu))\
           + loss_f(ev, torch.zeros_like(ev))
    return loss

def Outputs(X_batch):
    x = X_batch[:, 0:1]
    y = X_batch[:, 1:2]
    miu = X_batch[:, 2:3]

    x.requires_grad = True
    y.requires_grad = True
    miu.requires_grad = True

    input = torch.cat((x, y, miu), 1)
    output = net(input)
    p = output[:, 0:1]
    u = output[:, 1:2]
    v = output[:, 2:3]

    # exact boundary
    # p_hard = p
    # u_hard = u
    # v_hard = v
    p_hard = 0.1 * (1 - x) - x * (1 - x) * p
    u_hard = u * (1 - y ** 2 / 0.05 ** 2)
    v_hard = v

    u_x = torch.autograd.grad(u_hard, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u_hard, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v_hard, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v_hard, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
    p_x = torch.autograd.grad(p_hard, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
    p_y = torch.autograd.grad(p_hard, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]

    em = u_x + v_y
    eu = u_hard * u_x + v_hard * u_y + 1 / rho * p_x - miu / rho * (u_xx + u_yy)
    ev = u_hard * v_x + v_hard * v_y + 1 / rho * p_y - miu / rho * (v_xx + v_yy)


    return p_hard, u_hard, v_hard, u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, p_x, p_y, em, eu, ev

######dataset######
seed(13)
data = load_data()
X_tr, X_ts, y_tr, y_ts = train_test_split(data, miu_threshold=1.3e-4)
X_tr = X_tr.to(device)
X_ts = X_ts.to(device)
y_tr = y_tr.to(device)
y_ts = y_ts.to(device)
train_data = XyDataset(X_tr, y_tr)
test_data = XyDataset(X_ts, y_ts)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

######network######
net = nn.Sequential(nn.Linear(3, 100),
                    torch.nn.Tanh(),
                    nn.Linear(100, 300),
                    torch.nn.Tanh(),
                    nn.Linear(300, 100),
                    torch.nn.Tanh(),
                    nn.Linear(100, 3))
net = net.to(device)
optimizer = optim.Adam(net.parameters(),lr=1e-3)
mse = MSELoss()

######main######
for epoch in range(401):
    for i, ((X_train_batch, y_train_batch), (X_test_batch, y_test_batch)) in enumerate(zip(train_loader, test_loader)):
        optimizer.zero_grad()
        loss = loss_data(X_train_batch, y_train_batch) + loss_eqn(X_train_batch) + loss_eqn(X_test_batch)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 1:
        print("Epoch %03d: %f, %f, %f, %f" % (epoch, loss_data(X_tr,y_tr).item(), loss_data(X_ts,y_ts).item(), loss_eqn(X_tr).item(), loss_eqn(X_ts).item()))

torch.save(net.state_dict(), 'model_pipeflow.pth')
# net.load_state_dict(torch.load('model_pipeflow.pth'))
######save######
path = ""
for miu_value in [0.001, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1.5e-4, 1.2e-4, 1e-4, 9e-5, 8e-5, 7e-5]:
    x = data[miu_value]['x']
    y = data[miu_value]['y']
    u_true = data[miu_value]['u']
    v_true = data[miu_value]['v']
    p_true = data[miu_value]['p']
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    u_true = torch.Tensor(u_true).to(device)
    v_true = torch.Tensor(v_true).to(device)
    p_true = torch.Tensor(p_true).to(device)
    miu = miu_value * torch.ones(len(x), 1)
    miu = torch.Tensor(miu).to(device)
    input = torch.cat([x, y, miu], dim=1)
    output_tensors = Outputs(input)
    p_hard, u_hard, v_hard, u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, p_x, p_y, em, eu, ev = [tensor.detach().cpu().numpy() for
                                                                                                tensor in
                                                                                                output_tensors]
    # errors
    velocity_true = torch.sqrt(u_true**2 + v_true**2)
    velocity = np.sqrt(u_hard**2 + v_hard**2)
    velocity = torch.FloatTensor(velocity).to(device)
    p_hard_tensor = torch.FloatTensor(p_hard).to(device)
    print(nmae(p_hard_tensor, p_true).item(), nmae(velocity, velocity_true).item())
    # save
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    np.savez(path+'miu_'+str(miu_value)+'.npz',
             x=x, y=y,
             u=u_hard, v=v_hard, p=p_hard,
             u_x=u_x, u_xx=u_xx, u_y=u_y, u_yy=u_yy,
             v_x=v_x, v_xx=v_xx, v_y=v_y, v_yy=v_yy,
             p_x=p_x, p_y=p_y,
             em=em, eu=eu, ev=ev)








