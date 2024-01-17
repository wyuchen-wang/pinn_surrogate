import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import os
from torch import optim

# fluid parameters
miu = 0.00345
rio = 1050.
rio00 = 1050.
l00 = 0.003
u00 = 0.14
Re = rio*u00*l00/miu

# nmae error
def nmae(y_pred, y_true):
  mae = torch.mean(torch.abs(y_true - y_pred))
  normalization_factor = torch.max(y_true) - torch.min(y_true)
  return mae / normalization_factor

# Model 0.25pointnet
class PointNetEncoder(nn.Module):
  def __init__(self, channel=3):
    super(PointNetEncoder, self).__init__()
    self.conv1 = torch.nn.Conv1d(channel, 16, 1)
    self.conv2 = torch.nn.Conv1d(16, 32, 1)
    self.conv3 = torch.nn.Conv1d(32, 256, 1)
    self.bn1 = nn.BatchNorm1d(16)
    self.bn2 = nn.BatchNorm1d(32)
    self.bn3 = nn.BatchNorm1d(256)
    self.tanh = nn.Tanh()

  def forward(self, x):
    B, D, N = x.size()
    x = self.tanh(self.bn1(self.conv1(x)))
    pointfeat = x
    x = self.tanh(self.bn2(self.conv2(x)))
    x = self.bn3(self.conv3(x))
    x = torch.max(x, 2, keepdim=True)[0]
    x = x.view(-1, 256)
    x = x.view(-1, 256, 1).repeat(1, 1, N)
    return torch.cat([x, pointfeat], 1)

class PointNetPartSeg(nn.Module):
  def __init__(self, num_class=4):
    super(PointNetPartSeg, self).__init__()
    self.k = num_class
    self.feat = PointNetEncoder(channel=3)
    self.conv1 = torch.nn.Conv1d(272, 128, 1)
    self.conv2 = torch.nn.Conv1d(128, 64, 1)
    self.conv3 = torch.nn.Conv1d(64, 32, 1)
    self.conv4 = torch.nn.Conv1d(32, self.k, 1)
    self.bn1 = nn.BatchNorm1d(128)
    self.bn1_1 = nn.BatchNorm1d(1024)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(32)
    self.tanh = nn.Tanh()

  def forward(self, x):
    batchsize = x.size()[0]
    n_pts = x.size()[2]
    x = self.feat(x)
    x = self.tanh(self.bn1(self.conv1(x)))
    x = self.tanh(self.bn2(self.conv2(x)))
    x = self.tanh(self.bn3(self.conv3(x)))
    x = self.conv4(x)
    x = x.transpose(2, 1).contiguous()
    x = x.view(batchsize, n_pts, self.k)
    return x

model = PointNetPartSeg()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
criterion = torch.nn.MSELoss(reduction='mean')

def Loss_eqn(net_in):
  x = net_in[:, :, 0:1]
  y = net_in[:, :, 1:2]
  z = net_in[:, :, 2:3]
  x.requires_grad = True
  y.requires_grad = True
  z.requires_grad = True
  point_cloud_in = torch.cat([x, y, z], 2)
  point_cloud_in = point_cloud_in.transpose(2, 1).contiguous()
  puvw = model(point_cloud_in)
  p = puvw[:, :, 0:1]
  u = puvw[:, :, 1:2]
  v = puvw[:, :, 2:3]
  w = puvw[:, :, 3:4]
  p = p * rio00 * u00 * u00
  u = u * u00
  v = v * u00
  w = w * u00

  u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
  u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
  v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
  w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  loss_continuity = u_x + v_y + w_z
  loss_1 = u * u_x + v * u_y + w * u_z + 1/rio00 * p_x - miu/rio00 * (u_xx + u_yy + u_zz)
  loss_2 = u * v_x + v * v_y + w * v_z + 1/rio00 * p_y - miu/rio00 * (v_xx + v_yy + v_zz)
  loss_3 = u * w_x + v * w_y + w * w_z + 1/rio00 * p_z - miu/rio00 * (w_xx + w_yy + w_zz)

  loss_eqn = criterion(loss_1, torch.zeros_like(loss_1)) + criterion(loss_2, torch.zeros_like(loss_2)) + criterion(
    loss_3, torch.zeros_like(loss_3)) + \
             criterion(loss_continuity, torch.zeros_like(loss_continuity))
  return loss_eqn

def Loss_data(point_cloud,label):
    y_pred = model(point_cloud)
    loss_data = criterion(y_pred, label)
    return loss_data

def Loss_data_BC(point_cloud, label):
  y_pred = model(point_cloud)
  p_pred = y_pred[:,:,0:1]
  u_pred = y_pred[:,:,1:2]
  v_pred = y_pred[:,:,2:3]
  w_pred = y_pred[:,:,3:4]
  y_pred = torch.cat((u_pred, v_pred, w_pred), dim=2)
  loss_data = criterion(y_pred, label)
  return loss_data

def Output(net_in):
  x = net_in[:, :, 0:1]  # (2,4000,1)
  y = net_in[:, :, 1:2]
  z = net_in[:, :, 2:3]
  x.requires_grad = True
  y.requires_grad = True
  z.requires_grad = True
  point_cloud_in = torch.cat([x, y, z], 2)
  point_cloud_in = point_cloud_in.transpose(2, 1).contiguous()
  puvw = model(point_cloud_in)
  p = puvw[:, :, 0:1]
  u = puvw[:, :, 1:2]  # (2,4000,1)
  v = puvw[:, :, 2:3]
  w = puvw[:, :, 3:4]
  p = p * rio00 * u00 * u00
  u = u * u00
  v = v * u00
  w = w * u00
  outputs = torch.cat((p,u,v,w),dim=2)
  u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
  u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
  v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]
  w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
  p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
  p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

  loss_continuity = u_x + v_y + w_z
  loss_1 = u * u_x + v * u_y + w * u_z + 1 / rio00 * p_x - miu / rio00 * (u_xx + u_yy + u_zz)
  loss_2 = u * v_x + v * v_y + w * v_z + 1 / rio00 * p_y - miu / rio00 * (v_xx + v_yy + v_zz)
  loss_3 = u * w_x + v * w_y + w * w_z + 1 / rio00 * p_z - miu / rio00 * (w_xx + w_yy + w_zz)

  loss_eqn = criterion(loss_1, torch.zeros_like(loss_1)) + criterion(loss_2, torch.zeros_like(loss_2)) + criterion(
    loss_3, torch.zeros_like(loss_3)) + \
             criterion(loss_continuity, torch.zeros_like(loss_continuity))
  return outputs, u_x, u_xx, u_y, u_yy, u_z, u_zz, v_x, v_xx, v_y, v_yy, v_z, v_zz, w_x, w_xx, w_y, w_yy, w_z, w_zz,\
         p_x, p_y, p_z, loss_continuity, loss_1, loss_2, loss_3, loss_eqn

def Output_puvw(net_in):
  x = net_in[:, :, 0:1]  # (2,4000,1)
  y = net_in[:, :, 1:2]
  z = net_in[:, :, 2:3]
  x.requires_grad = True
  y.requires_grad = True
  z.requires_grad = True
  point_cloud_in = torch.cat([x, y, z], 2)
  point_cloud_in = point_cloud_in.transpose(2, 1).contiguous()
  puvw = model(point_cloud_in)
  p = puvw[:, :, 0:1]
  u = puvw[:, :, 1:2]  # (2,4000,1)
  v = puvw[:, :, 2:3]
  w = puvw[:, :, 3:4]
  p = p * rio00 * u00 * u00
  u = u * u00
  v = v * u00
  w = w * u00
  outputs = torch.cat((p,u,v,w),dim=2)

  return outputs


if __name__=="__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  file_folder = ''
  ######train files######
  file_train_list = ['mesh15e-5/1515.npz', 'mesh15e-5/1520.npz', 'mesh15e-5/1530.npz', 'mesh15e-5/1540.npz',
               'mesh15e-5/2015.npz', 'mesh15e-5/2020.npz', 'mesh15e-5/2030.npz', 'mesh15e-5/2040.npz',
               'mesh15e-5/3015.npz', 'mesh15e-5/3020.npz', 'mesh15e-5/3030.npz', 'mesh15e-5/3040.npz',
               'mesh15e-5/4015.npz', 'mesh15e-5/4020.npz', 'mesh15e-5/4030.npz', 'mesh15e-5/4040.npz']
  X_train_list = []
  P_train_list = []
  U_train_list = []
  V_train_list = []
  W_train_list = []
  for file_name in file_train_list:
      data = np.load(file_name)
      x = data['x']
      y = data['y']
      z = data['z']
      p = data['p']
      u = data['u']
      v = data['v']
      w = data['w']
      X = np.concatenate((x,y,z), axis=1)
      X_train_list.append(X)
      P_train_list.append(p)
      U_train_list.append(u)
      V_train_list.append(v)
      W_train_list.append(w)
  # normalization
  P_train_list = [[num/(rio00*u00*u00) for num in sublist] for sublist in P_train_list]
  U_train_list = [[num/u00 for num in sublist] for sublist in U_train_list]
  V_train_list = [[num/u00 for num in sublist] for sublist in V_train_list]
  W_train_list = [[num/u00 for num in sublist] for sublist in W_train_list]
  ######test files######
  file_test_list = [
               'mesh15e-5/1550.npz', 'mesh15e-5/1560.npz', 'mesh15e-5/1570.npz',
               'mesh15e-5/2050.npz', 'mesh15e-5/2060.npz', 'mesh15e-5/2070.npz', 
               'mesh15e-5/3050.npz', 'mesh15e-5/3060.npz', 'mesh15e-5/3070.npz',
               'mesh15e-5/4050.npz', 'mesh15e-5/4060.npz', 'mesh15e-5/4070.npz', 
               'mesh15e-5/5015.npz', 'mesh15e-5/5020.npz', 'mesh15e-5/5030.npz', 'mesh15e-5/5040.npz', 'mesh15e-5/5050.npz',
               'mesh15e-5/5060.npz', 'mesh15e-5/5070.npz', 
               'mesh15e-5/6015.npz', 'mesh15e-5/6020.npz', 'mesh15e-5/6030.npz', 'mesh15e-5/6040.npz', 'mesh15e-5/6050.npz',
               'mesh15e-5/6060.npz', 'mesh15e-5/6070.npz',
               'mesh15e-5/7015.npz', 'mesh15e-5/7020.npz', 'mesh15e-5/7030.npz', 'mesh15e-5/7040.npz', 'mesh15e-5/7050.npz', 
               'mesh15e-5/7060.npz', 'mesh15e-5/7070.npz']
  X_test_list = []
  for file_name in file_test_list:
      data = np.load(file_name)
      x = data['x']
      y = data['y']
      z = data['z']
      X = np.concatenate((x, y, z), axis=1)
      X_test_list.append(X)
  ######test files BC######
  file_testBC_list = [
    'inlet+wall_BC_npz/1550.npz', 'inlet+wall_BC_npz/1560.npz', 'inlet+wall_BC_npz/1570.npz',
    'inlet+wall_BC_npz/2050.npz', 'inlet+wall_BC_npz/2060.npz', 'inlet+wall_BC_npz/2070.npz',
    'inlet+wall_BC_npz/3050.npz', 'inlet+wall_BC_npz/3060.npz', 'inlet+wall_BC_npz/3070.npz',
    'inlet+wall_BC_npz/4050.npz', 'inlet+wall_BC_npz/4060.npz', 'inlet+wall_BC_npz/4070.npz',
    'inlet+wall_BC_npz/5015.npz', 'inlet+wall_BC_npz/5020.npz', 'inlet+wall_BC_npz/5030.npz',
    'inlet+wall_BC_npz/5040.npz', 'inlet+wall_BC_npz/5050.npz',
    'inlet+wall_BC_npz/5060.npz', 'inlet+wall_BC_npz/5070.npz',
    'inlet+wall_BC_npz/6015.npz', 'inlet+wall_BC_npz/6020.npz', 'inlet+wall_BC_npz/6030.npz',
    'inlet+wall_BC_npz/6040.npz', 'inlet+wall_BC_npz/6050.npz',
    'inlet+wall_BC_npz/6060.npz', 'inlet+wall_BC_npz/6070.npz',
    'inlet+wall_BC_npz/7015.npz', 'inlet+wall_BC_npz/7020.npz', 'inlet+wall_BC_npz/7030.npz',
    'inlet+wall_BC_npz/7040.npz', 'inlet+wall_BC_npz/7050.npz',
    'inlet+wall_BC_npz/7060.npz', 'inlet+wall_BC_npz/7070.npz']
  X_testBC_list = []
  U_testBC_list = []
  V_testBC_list = []
  W_testBC_list = []
  for file_name in file_testBC_list:
      data = np.load(file_name)
      x = data['x']
      y = data['y']
      z = data['z']
      u = data['u']
      v = data['v']
      w = data['w']
      X = np.concatenate((x, y, z), axis=1)
      X_testBC_list.append(X)
      U_testBC_list.append(u)
      V_testBC_list.append(v)
      W_testBC_list.append(w)
  # normalization
  U_testBC_list = [[num / u00 for num in sublist] for sublist in U_testBC_list]
  V_testBC_list = [[num / u00 for num in sublist] for sublist in V_testBC_list]
  W_testBC_list = [[num / u00 for num in sublist] for sublist in W_testBC_list]
  ######all files######
  file_all_list = ['mesh15e-5/1515.npz', 'mesh15e-5/1520.npz', 'mesh15e-5/1530.npz', 'mesh15e-5/1540.npz',
                   'mesh15e-5/1550.npz', 'mesh15e-5/1560.npz', 'mesh15e-5/1570.npz',
                   'mesh15e-5/2015.npz', 'mesh15e-5/2020.npz', 'mesh15e-5/2030.npz', 'mesh15e-5/2040.npz',
                   'mesh15e-5/2050.npz', 'mesh15e-5/2060.npz', 'mesh15e-5/2070.npz',
                   'mesh15e-5/3015.npz', 'mesh15e-5/3020.npz', 'mesh15e-5/3030.npz', 'mesh15e-5/3040.npz',
                   'mesh15e-5/3050.npz', 'mesh15e-5/3060.npz', 'mesh15e-5/3070.npz',
                   'mesh15e-5/4015.npz', 'mesh15e-5/4020.npz', 'mesh15e-5/4030.npz', 'mesh15e-5/4040.npz',
                   'mesh15e-5/4050.npz', 'mesh15e-5/4060.npz', 'mesh15e-5/4070.npz',
                   'mesh15e-5/5015.npz', 'mesh15e-5/5020.npz', 'mesh15e-5/5030.npz', 'mesh15e-5/5040.npz',
                   'mesh15e-5/5050.npz', 'mesh15e-5/5060.npz', 'mesh15e-5/5070.npz',
                   'mesh15e-5/6015.npz', 'mesh15e-5/6020.npz', 'mesh15e-5/6030.npz', 'mesh15e-5/6040.npz',
                   'mesh15e-5/6050.npz', 'mesh15e-5/6060.npz', 'mesh15e-5/6070.npz',
                   'mesh15e-5/7015.npz', 'mesh15e-5/7020.npz', 'mesh15e-5/7030.npz', 'mesh15e-5/7040.npz',
                   'mesh15e-5/7050.npz', 'mesh15e-5/7060.npz', 'mesh15e-5/7070.npz']
  ######main######
  for iterations in range(100001):
    #idx
    idx = np.random.choice(16, 1, replace=False)
    idx = idx[0]
    Nbif = 2000
    # loss_data of train set
    X = X_train_list[idx]
    p = P_train_list[idx]
    u = U_train_list[idx]
    v = V_train_list[idx]
    w = W_train_list[idx]
    X = np.array([X])
    p = np.array([p])
    u = np.array([u])
    v = np.array([v])
    w = np.array([w])
    len = X.shape[1]
    iddxx = np.random.choice(len, Nbif, replace=False)
    X = X[:, iddxx, 0:3]
    p = p[:, iddxx, 0:1]
    u = u[:, iddxx, 0:1]
    v = v[:, iddxx, 0:1]
    w = w[:, iddxx, 0:1]
    X = torch.from_numpy(X).to(torch.float32)
    p = torch.from_numpy(p).to(torch.float32)
    u = torch.from_numpy(u).to(torch.float32)
    v = torch.from_numpy(v).to(torch.float32)
    w = torch.from_numpy(w).to(torch.float32)
    X = X.transpose(2, 1).contiguous()
    point_cloud = torch.cat([X,X], dim=0)
    puvw = torch.cat([p,u,v,w], dim=2)
    label = torch.cat([puvw,puvw], dim=0)
    point_cloud = point_cloud.to(device)
    label = label.to(device)
    loss_data = Loss_data(point_cloud, label)
    # loss_eqn of train set
    xin = X_train_list[idx]
    xin = np.array([xin])
    len = xin.shape[1]
    xin = xin[:, iddxx, 0:3]
    xin = torch.from_numpy(xin).to(torch.float32)
    net_in = torch.cat([xin, xin],dim=0) #(2,4000,3)
    net_in = net_in.to(device)
    loss_eqn = Loss_eqn(net_in)
    # loss_test_eqn
    idx_test = np.random.choice(33, 1, replace=False)
    idx_test = idx_test[0]
    Xtest = X_test_list[idx_test]
    Xtest = np.array([Xtest])
    len = Xtest.shape[1]
    iddxx_test = np.random.choice(len, Nbif, replace=False)
    Xtest = Xtest[:, iddxx_test, 0:3]
    Xtest = torch.from_numpy(Xtest).to(torch.float32)
    net_in_test = torch.cat([Xtest, Xtest], dim=0)
    net_in_test = net_in_test.to(device)
    loss_eqn_test = Loss_eqn(net_in_test)
    # test BC
    XtestBC = X_testBC_list[idx_test] #the same file
    utestBC = U_testBC_list[idx_test]
    vtestBC = V_testBC_list[idx_test]
    wtestBC = W_testBC_list[idx_test]
    XtestBC = np.array([XtestBC])
    utestBC = np.array([utestBC])
    vtestBC = np.array([vtestBC])
    wtestBC = np.array([wtestBC])
    len = XtestBC.shape[1]
    iddxx_BC = np.random.choice(len, Nbif, replace=False)
    XtestBC = XtestBC[:, iddxx_BC, 0:3]
    utestBC = utestBC[:, iddxx_BC, 0:1]
    vtestBC = vtestBC[:, iddxx_BC, 0:1]
    wtestBC = wtestBC[:, iddxx_BC, 0:1]
    XtestBC = torch.from_numpy(XtestBC).to(torch.float32)
    utestBC = torch.from_numpy(utestBC).to(torch.float32)
    vtestBC = torch.from_numpy(vtestBC).to(torch.float32)
    wtestBC = torch.from_numpy(wtestBC).to(torch.float32)
    XtestBC = XtestBC.transpose(2, 1).contiguous()
    point_cloud = torch.cat([XtestBC, XtestBC], dim=0)
    uvwtestBC = torch.cat([utestBC, vtestBC, wtestBC], dim=2)
    label = torch.cat([uvwtestBC, uvwtestBC], dim=0)
    point_cloud = point_cloud.to(device)
    label = label.to(device)
    loss_data_testBC = Loss_data_BC(point_cloud, label)

    optimizer.zero_grad()
    loss = loss_data + 0.05*loss_eqn + 0.05*loss_eqn_test + loss_data_testBC
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
       print(epoch, loss_data.item(), loss_eqn.item(), loss_eqn_test.item(), loss_data_testBC.item())

  torch.save(model.state_dict(), 'model_bifurcating_data+0.05pinn+0.05pinn+testbc.pth')
  ######outputs######
  for file_name in file_all_list:
      data = np.load(file_name)
      name = file_name.split('/')[1].split('.')[0]
      x = data['x']
      y = data['y']
      z = data['z']
      p = data['p']
      u = data['u']
      v = data['v']
      w = data['w']
      X = np.concatenate((x, y, z), axis=1)
      uvw = np.concatenate((u, v, w), axis=1)
      velocity = np.sqrt(u ** 2 + v ** 2 + w ** 2)
      # # samples
      # N = X.shape[0]
      # iddxx = np.random.choice(N, 4000, replace=False)
      # X = data[iddxx, 0:3]
      # p = data[iddxx, 3:4]
      # uvw = data[iddxx, 4:7]
      # u = data[iddxx, 4:5]
      # v = data[iddxx, 5:6]
      # w = data[iddxx, 6:7]
      # velocity = np.sqrt(u ** 2 + v ** 2 + w ** 2)
      xin = np.array([X])
      xin = torch.Tensor(xin).to(device)
      net_in = torch.cat([xin, xin], dim=0)  # (2,N,3)
      net_in = net_in.to(device)
      # outputs, u_x, u_xx, u_y, u_yy, u_z, u_zz, v_x, v_xx, v_y, v_yy, v_z, v_zz, w_x, w_xx, w_y, w_yy, w_z, w_zz, \
      # p_x, p_y, p_z, loss_continuity, loss_1, loss_2, loss_3, loss_eqn = Output(net_in)
      outputs = Output_puvw(net_in)
      y_p = outputs[:, :, 0:1].detach()
      y_u = outputs[:, :, 1:2].detach()
      y_v = outputs[:, :, 2:3].detach()
      y_w = outputs[:, :, 3:4].detach()
      # y_p = y_p * rio00 * u00 * u00
      # y_u = y_u * u00
      # y_v = y_v * u00
      # y_w = y_w * u00
      y_uvw = torch.cat([y_u, y_v, y_w], dim=2)
      y_velocity = torch.sqrt(y_u ** 2 + y_v ** 2 + y_w ** 2)
      # error
      label_p = np.array([p, p])
      label_uvw = np.array([uvw, uvw])
      label_p = torch.Tensor(label_p).to(device)
      label_uvw = torch.Tensor(label_uvw).to(device)
      label_velocity = np.array([velocity])
      label_velocity = torch.Tensor(label_velocity).to(device)
      loss_p = criterion(y_p, label_p)
      loss_uvw = criterion(y_uvw, label_uvw)
      nmae_p = nmae(y_p, label_p)
      nmae_velocity = nmae(y_velocity, label_velocity)
      print(loss_p.item(), loss_uvw.item(), nmae_p.item(), nmae_velocity.item())
      # save
      x_save = X[:, 0:1]
      y_save = X[:, 1:2]
      z_save = X[:, 2:3]
      y_p_save = y_p[0:1, :, :].squeeze(0).cpu().numpy()
      y_u_save = y_u[0:1, :, :].squeeze(0).cpu().numpy()
      y_v_save = y_v[0:1, :, :].squeeze(0).cpu().numpy()
      y_w_save = y_w[0:1, :, :].squeeze(0).cpu().numpy()
      # u_x_save = u_x[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # u_xx_save = u_xx[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # u_y_save = u_y[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # u_yy_save = u_yy[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # u_z_save = u_z[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # u_zz_save = u_zz[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # v_x_save = v_x[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # v_xx_save = v_xx[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # v_y_save = v_y[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # v_yy_save = v_yy[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # v_z_save = v_z[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # v_zz_save = v_zz[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # w_x_save = w_x[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # w_xx_save = w_xx[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # w_y_save = w_y[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # w_yy_save = w_yy[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # w_z_save = w_z[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # w_zz_save = w_zz[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # p_x_save = p_x[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # p_y_save = p_y[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # p_z_save = p_z[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # loss_continuity_save = loss_continuity[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # loss_1_save = loss_1[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # loss_2_save = loss_2[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # loss_3_save = loss_3[0:1, :, :].squeeze(0).detach().cpu().numpy()
      # np.savez(file_folder + name + '.npz', x=x_save, y=y_save, z=z_save, p=y_p_save, u=y_u_save, v=y_v_save, w=y_w_save,
      #          u_x=u_x_save, u_xx=u_xx_save, u_y=u_y_save, u_yy=u_yy_save, u_z=u_z_save, u_zz=u_zz_save,
      #          v_x=v_x_save, v_xx=v_xx_save, v_y=v_y_save, v_yy=v_yy_save, v_z=v_z_save, v_zz=v_zz_save,
      #          w_x=w_x_save, w_xx=w_xx_save, w_y=w_y_save, w_yy=w_yy_save, w_z=w_z_save, w_zz=w_zz_save,
      #          p_x=p_x_save, p_y=p_y_save, p_z=p_z_save,
      #          loss_continuity=loss_continuity_save, loss_1=loss_1_save, loss_2=loss_2_save, loss_3=loss_3_save)
      np.savez(file_folder + name + '.npz', x=x_save, y=y_save, z=z_save, p=y_p_save, u=y_u_save, v=y_v_save,
               w=y_w_save)