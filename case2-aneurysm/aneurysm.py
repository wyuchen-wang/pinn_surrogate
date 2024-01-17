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

# fluid parameters
miu = 0.001
rho = 1.
pi_tensor = torch.tensor(math.pi)
nu = 1e-3

# nmae errors
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

def load_data():
    data = {}

    for fn in os.listdir("aneurysm"):
        A = float(fn[1:fn.index("n")-1])
        data[A] = np.load('aneurysm/' + fn)

    return data

class XyDataset(Dataset):
    def __init__(self, X, y):
      self.X, self.y = X, y

    def __len__(self):
      return len(self.X)

    def __getitem__(self, idx):
      return self.X[idx], self.y[idx]

mse = MSELoss()

def loss_data(X_batch, y_batch):
    x = X_batch[:, 0:1]
    y = X_batch[:, 1:2]
    A = X_batch[:, 2:3]

    input = torch.cat((x, y, A), 1)
    output = net(input)
    p = output[:, 0:1]
    u = output[:, 1:2]
    v = output[:, 2:3]

    # exact boundary
    sigma = 0.1
    mu = 0.5
    rInlet = 0.05
    xStart = 0
    dP = 0.1
    xEnd = 1
    L = 1

    R = A * 1 / torch.sqrt(2 * torch.tensor(np.pi) * sigma ** 2) * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)).to(
        device)
    h = rInlet - R
    h = h.to(device)

    u_hard = u * (1 - y ** 2 / h ** 2)
    v_hard = v
    p_hard = (xStart - x) * 0 + dP * (xEnd - x) / L + 0 * y + (xStart - x) * (xEnd - x) * p
    # u_hard = u
    # v_hard = v
    # p_hard = p
    output_cal = torch.cat([p_hard, u_hard, v_hard], dim=1)
    loss_data = mse(output_cal, y_batch)
    return loss_data

def loss_eqn(X_batch):
    x = X_batch[:,0:1]
    y = X_batch[:,1:2]
    A = X_batch[:,2:3]

    x.requires_grad = True
    y.requires_grad = True
    A.requires_grad = True

    input = torch.cat((x, y, A), 1)
    output = net(input)
    p = output[:,0:1]
    u = output[:,1:2]
    v = output[:,2:3]

    # exact boundary
    sigma = 0.1
    mu = 0.5
    rInlet = 0.05
    xStart = 0
    dP = 0.1
    xEnd = 1
    L = 1

    R = A * 1 / torch.sqrt(2 * torch.tensor(np.pi) * sigma ** 2) * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)).to(
        device)
    h = rInlet - R
    h = h.to(device)

    u_hard = u * (1 - y ** 2 / h ** 2)
    v_hard = v
    p_hard = (xStart - x) * 0 + dP * (xEnd - x) / L + 0 * y + (xStart - x) * (xEnd - x) * p
    # u_hard = u
    # v_hard = v
    # p_hard = p

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

    eu = (u_hard * u_x + v_hard * u_y - miu/rho * (u_xx + u_yy) + 1 / rho * p_x)
    ev = (u_hard * v_x + v_hard * v_y - miu/rho * (v_xx + v_yy) + 1 / rho * p_y)
    em = (u_x + v_y)


    # MSE LOSS
    loss_f = nn.MSELoss()

    loss = loss_f(eu, torch.zeros_like(eu)) \
           + loss_f(ev, torch.zeros_like(ev)) \
           + loss_f(em, torch.zeros_like(em))
    return loss

###Model###
net = nn.Sequential(
    nn.Linear(3, 100),
    torch.nn.Tanh(),
    nn.Linear(100, 300),
    torch.nn.Tanh(),
    nn.Linear(300, 100),
    torch.nn.Tanh(),
    nn.Linear(100, 3)
)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net = net.to(device)
optimizer = optim.Adam(net.parameters(),lr=1.e-3)

if __name__=="__main__":
  seed(1)
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  ###dataset prepare###
  file_list = ['aneurysm/A0.0.npz','aneurysm/A-0.0004.npz','aneurysm/A-0.0008.npz','aneurysm/A-0.0012.npz','aneurysm/A-0.0016.npz',
               'aneurysm/A-0.0020.npz','aneurysm/A-0.0024.npz','aneurysm/A-0.0028.npz','aneurysm/A-0.0032.npz','aneurysm/A-0.0036.npz',
               'aneurysm/A-0.0040.npz','aneurysm/A-0.0044.npz','aneurysm/A-0.0048.npz','aneurysm/A-0.0052.npz','aneurysm/A-0.0056.npz',
               'aneurysm/A-0.0060.npz','aneurysm/A-0.0064.npz','aneurysm/A-0.0068.npz','aneurysm/A-0.0072.npz','aneurysm/A-0.0076.npz',
               'aneurysm/A-0.0080.npz','aneurysm/A-0.0084.npz','aneurysm/A-0.0088.npz','aneurysm/A-0.0092.npz','aneurysm/A-0.0096.npz',
               'aneurysm/A-0.0100.npz','aneurysm/A-0.0104.npz','aneurysm/A-0.0108.npz','aneurysm/A-0.0112.npz','aneurysm/A-0.0116.npz',
               'aneurysm/A-0.0120.npz','aneurysm/A-0.0124.npz','aneurysm/A-0.0128.npz','aneurysm/A-0.0132.npz','aneurysm/A-0.0136.npz',
               'aneurysm/A-0.0140.npz','aneurysm/A-0.0144.npz','aneurysm/A-0.0148.npz','aneurysm/A-0.0152.npz','aneurysm/A-0.0156.npz',
               'aneurysm/A-0.0160.npz','aneurysm/A-0.0164.npz','aneurysm/A-0.0168.npz','aneurysm/A-0.0172.npz','aneurysm/A-0.0176.npz',
               'aneurysm/A-0.0180.npz','aneurysm/A-0.0184.npz','aneurysm/A-0.0188.npz','aneurysm/A-0.0192.npz','aneurysm/A-0.0196.npz',
               'aneurysm/A-0.0200.npz']
  X_list = []
  P_list = []
  U_list = []
  V_list = []
  for file_name in file_list:
    with np.load(file_name) as data:
      x = data['x_center']
      y = data['y_center']
      u = data['u_center']
      v = data['v_center']
      p = data['p_center']
      X = np.concatenate((x, y), axis=1)
      X_list.append(X)
      P_list.append(p)
      U_list.append(u)
      V_list.append(v)
  file_train_list = ['aneurysm/A0.0.npz', 'aneurysm/A-0.0004.npz', 'aneurysm/A-0.0008.npz', 'aneurysm/A-0.0012.npz',
               'aneurysm/A-0.0016.npz', 'aneurysm/A-0.0020.npz', 'aneurysm/A-0.0024.npz', 'aneurysm/A-0.0028.npz',
               'aneurysm/A-0.0032.npz', 'aneurysm/A-0.0036.npz', 'aneurysm/A-0.0040.npz', 'aneurysm/A-0.0044.npz',
               'aneurysm/A-0.0048.npz', 'aneurysm/A-0.0052.npz', 'aneurysm/A-0.0056.npz', 'aneurysm/A-0.0060.npz',
               'aneurysm/A-0.0064.npz', 'aneurysm/A-0.0068.npz', 'aneurysm/A-0.0072.npz', 'aneurysm/A-0.0076.npz']
  X_train_list = []
  y_train_list = []
  for file_name in file_train_list:
    with np.load(file_name) as data:
      x = data['x_center']
      y = data['y_center']
      u = data['u_center']
      v = data['v_center']
      p = data['p_center']
      # A
      N = len(x)
      A_start = file_name.index("A") + 1
      A_end = file_name.index(".n")
      A_value = float(file_name[A_start: A_end])
      input = np.concatenate([x, y, A_value * np.ones((N, 1))], axis=1)
      output = np.concatenate([p, u, v], axis=1)
      X_train_list.append(input)
      y_train_list.append(output)
  file_test_list = ['aneurysm/A-0.0080.npz', 'aneurysm/A-0.0084.npz', 'aneurysm/A-0.0088.npz', 'aneurysm/A-0.0092.npz',
                    'aneurysm/A-0.0096.npz', 'aneurysm/A-0.0100.npz', 'aneurysm/A-0.0104.npz', 'aneurysm/A-0.0108.npz',
                    'aneurysm/A-0.0112.npz', 'aneurysm/A-0.0116.npz', 'aneurysm/A-0.0120.npz', 'aneurysm/A-0.0124.npz',
                    'aneurysm/A-0.0128.npz', 'aneurysm/A-0.0132.npz', 'aneurysm/A-0.0136.npz', 'aneurysm/A-0.0140.npz',
                    'aneurysm/A-0.0144.npz', 'aneurysm/A-0.0148.npz', 'aneurysm/A-0.0152.npz', 'aneurysm/A-0.0156.npz',
                    'aneurysm/A-0.0160.npz', 'aneurysm/A-0.0164.npz', 'aneurysm/A-0.0168.npz', 'aneurysm/A-0.0172.npz',
                    'aneurysm/A-0.0176.npz', 'aneurysm/A-0.0180.npz', 'aneurysm/A-0.0184.npz', 'aneurysm/A-0.0188.npz',
                    'aneurysm/A-0.0192.npz', 'aneurysm/A-0.0196.npz', 'aneurysm/A-0.0200.npz']
  X_test_list = []
  y_test_list = []
  for file_name in file_test_list:
    with np.load(file_name) as data:
      x = data['x_center']
      y = data['y_center']
      u = data['u_center']
      v = data['v_center']
      p = data['p_center']
      # A
      N = len(x)
      A_start = file_name.index("A") + 1
      A_end = file_name.index(".n")
      A_value = float(file_name[A_start: A_end])
      input = np.concatenate([x, y, A_value * np.ones((N, 1))], axis=1)
      output = np.concatenate([p, u, v], axis=1)
      X_test_list.append(input)
      y_test_list.append(output)
  X_train = np.concatenate(X_train_list, axis=0)
  y_train = np.concatenate(y_train_list, axis=0)
  X_train = torch.FloatTensor(X_train)
  y_train = torch.FloatTensor(y_train)
  X_train = X_train.to(device)
  y_train = y_train.to(device)
  X_test = np.concatenate(X_test_list, axis=0)
  y_test = np.concatenate(y_test_list, axis=0)
  X_test = torch.FloatTensor(X_test)
  y_test = torch.FloatTensor(y_test)
  X_test = X_test.to(device)
  y_test = y_test.to(device)
  # dataloader
  train_data = XyDataset(X_train, y_train)
  test_data = XyDataset(X_test, y_test)
  train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

  ###main###
  for epoch in range(401):
    for i, ((X_train_batch, y_train_batch), (X_test_batch, y_test_batch)) in enumerate(zip(train_loader, test_loader)):
      optimizer.zero_grad()
      loss = loss_data(X_train_batch, y_train_batch)/loss_data(X_train_batch, y_train_batch).detach() \
             + criterion(X_train_batch)/criterion(X_train_batch).detach() \
             + criterion(X_test_batch)/criterion(X_test_batch).detach()
      # loss = loss_data(X_train_batch, y_train_batch) \
      #       + loss_eqn(X_train_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 1:
      print("Epoch %03d: %f, %f, %f, %f" % (epoch, loss_data(X_train, y_train).item(), loss_data(X_test, y_test).item(), loss_eqn(X_train_batch), loss_eqn(X_test_batch)))

  torch.save(net.state_dict(), 'net.pth')

  ###outputs###
  path = ""
  for idx in range(51):
    # dataset
    idx_co = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,
              24, 23, 22,
              21, 20, 19, 18,
              17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    scale_co = [-0.02, -0.0196, -0.0192, -0.0188, -0.0184, -0.018, -0.0176, -0.0172, -0.0168,
                -0.0164, -0.016, -0.0156, -0.0152, -0.0148, -0.0144, -0.014, -0.0136, -0.0132,
                -0.0128, -0.0124, -0.012, -0.0116, -0.0112, -0.0108, -0.0104, -0.01, -0.0096,
                -0.0092, -0.0088, -0.0084, -0.008, -0.0076, -0.0072, -0.0068, -0.0064, -0.006,
                -0.0056, -0.0052, -0.0048, -0.0044, -0.004, -0.0036, -0.0032, -0.0028, -0.0024,
                -0.002, -0.0016, -0.0012, -0.0008, -0.0004, 0]  # 总共51个数
    index = idx_co.index(idx)
    A_value = scale_co[index]
    # predict
    x = X_list[idx]
    p = P_list[idx]
    u = U_list[idx]
    v = V_list[idx]
    x = np.array(x)
    xx = x[:, 0:1]
    yy = x[:, 1:2]
    label_p = np.array(p)
    label_u = np.array(u)
    label_v = np.array(v)
    xx = torch.from_numpy(xx).to(torch.float32)
    yy = torch.from_numpy(yy).to(torch.float32)
    A = torch.full_like(xx, A_value)
    label_p = torch.from_numpy(label_p).to(torch.float32)
    label_u = torch.from_numpy(label_u).to(torch.float32)
    label_v = torch.from_numpy(label_v).to(torch.float32)
    label_p = label_p.to(device)
    label_u = label_u.to(device)
    label_v = label_v.to(device)
    xx = xx.to(device)
    yy = yy.to(device)
    A = A.to(device)
    xx.requires_grad = True
    yy.requires_grad = True
    A.requires_grad = True
    input = torch.cat([xx, yy, A], dim=1)
    output = net(input)
    p = output[:, 0:1]
    u = output[:, 1:2]
    v = output[:, 2:3]
    # exact boundary
    sigma = 0.1
    mu = 0.5
    rInlet = 0.05
    xStart = 0
    dP = 0.1
    xEnd = 1
    L = 1
    nu = 1e-3
    rho = 1
    R = A * 1 / torch.sqrt(2 * torch.tensor(np.pi) * sigma ** 2) * torch.exp(-(xx - mu) ** 2 / (2 * sigma ** 2)).to(
        device)
    h = rInlet - R
    h = h.to(device)
    u_hard = u * (1 - yy ** 2 / h ** 2)
    v_hard = v
    p_hard = 0.1 * (1 - xx) + (-xx) * (1 - xx) * p
    # u_hard = u
    # v_hard = v
    # p_hard = p
    # derivatives
    u_x = torch.autograd.grad(u_hard, xx, grad_outputs=torch.ones_like(xx), create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, xx, grad_outputs=torch.ones_like(xx), create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u_hard, yy, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, yy, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    v_x = torch.autograd.grad(v_hard, xx, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, xx, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v_hard, yy, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, yy, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    p_x = torch.autograd.grad(p_hard, xx, grad_outputs=torch.ones_like(xx), create_graph=True, only_inputs=True)[0]
    p_y = torch.autograd.grad(p_hard, yy, grad_outputs=torch.ones_like(yy), create_graph=True, only_inputs=True)[0]
    eu = (u_hard * u_x + v_hard * u_y - miu * (u_xx + u_yy) + 1 / rho * p_x)
    ev = (u_hard * v_x + v_hard * v_y - miu * (v_xx + v_yy) + 1 / rho * p_y)
    em = (u_x + v_y)
    # save
    puv_pred = torch.cat([p_hard, u_hard, v_hard],dim=1)
    velocity_pred = torch.sqrt(u_hard ** 2 + v_hard ** 2)
    xx_hat = xx.detach().cpu().numpy()
    yy_hat = yy.detach().cpu().numpy()
    A_hat = A.detach().cpu().numpy()
    p_hat = p_hard.detach().cpu().numpy()
    u_hat = u_hard.detach().cpu().numpy()
    v_hat = v_hard.detach().cpu().numpy()
    u_x_hat = u_x.detach().cpu().numpy()
    u_xx_hat = u_xx.detach().cpu().numpy()
    u_y_hat = u_y.detach().cpu().numpy()
    u_yy_hat = u_yy.detach().cpu().numpy()
    v_x_hat = v_x.detach().cpu().numpy()
    v_xx_hat = v_xx.detach().cpu().numpy()
    v_y_hat = v_y.detach().cpu().numpy()
    v_yy_hat = v_yy.detach().cpu().numpy()
    p_x_hat = p_x.detach().cpu().numpy()
    p_y_hat = p_y.detach().cpu().numpy()
    eu_hat = eu.detach().cpu().numpy()
    ev_hat = ev.detach().cpu().numpy()
    em_hat = em.detach().cpu().numpy()
    np.savez(path + 'A' + str(idx) + '.npz', x = xx_hat, y = yy_hat, A = A_hat, u = u_hat, v = v_hat, p = p_hat,
             u_x = u_x_hat, u_xx = u_xx_hat, u_y = u_y_hat, u_yy = u_yy_hat, v_x = v_x_hat, v_xx = v_xx_hat, 
             v_y = v_y_hat, v_yy = v_yy_hat, P_x = p_x_hat, P_y = p_y_hat, eu = eu_hat, ev = ev_hat,
             em = em_hat)
    # errors
    label_velocity = torch.sqrt(label_u ** 2 + label_v ** 2).to(device)
    loss_f = nn.MSELoss()
    loss_p = loss_f(p_hard, label_p)
    loss_u = loss_f(u_hard, label_u)
    loss_v = loss_f(v_hard, label_v)
    loss_velocity = loss_f(velocity_pred, label_velocity)
    nmae_p = nmae(p_hard, label_p)
    nmae_velocity = nmae(velocity_pred, label_velocity)
    print(loss_p.item(), loss_u.item(), loss_v.item(), loss_velocity.item(), nmae_p.item(), nmae_velocity.item())






