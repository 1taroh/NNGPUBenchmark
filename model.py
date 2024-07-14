import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size=32*32*3, hidden_size=512, output_size=10):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)

        return x
    
    def train_loop(self, trainloader, device, loss_fn, optimizer, batch_size=None) -> None:
        for batch in tqdm(trainloader):
            image, label = batch[0], batch[1]
            image = image.reshape(len(image), -1)

            # GPU に移行
            image = image.to(device)
            label = label.to(device)

            # 勾配を0で初期化（これをしないと逆伝播で計算した勾配が足されていく）
            optimizer.zero_grad()

            # 順伝播
            pred = self.forward(image)

            # ロスの計算
            loss = loss_fn(pred, label)

            # 逆伝播
            loss.backward()

            # パラメータの更新
            optimizer.step()


    def test_loop(self, testloader, device) -> list:
        with torch.no_grad():
            acc_list = []
            for batch in tqdm(testloader):
                image, label = batch[0], batch[1]
                image = image.reshape(len(image), -1)
                image = image.to(device)
                label = label.to(device)
                pred = self.forward(image)
                accuracy = 100 * torch.sum(torch.argmax(pred, dim=1) == label) / len(pred)
                acc_list.append(float(accuracy))
            print(f'Accuracy {np.mean(acc_list):.2f}')

        return acc_list

class CNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1        = nn.Conv2d(3,6,(5,5),1,0)#stride=2ではない？
      self.subsumpling2 = nn.MaxPool2d(2,2)
      self.conv3        = nn.Conv2d(6,16,(5,5),1,0)
      self.subsumpling4 = nn.MaxPool2d(2,2)
      self.l5           = nn.Linear(16*5*5,120)
      self.l6           = nn.Linear(120,84)
      self.output       = nn.Linear(84,10)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.subsumpling2(x)
      x = self.conv3(x)
      x = F.relu(x)
      x = self.subsumpling4(x)
      x = x.view(x.size()[0], -1)
      x = self.l5(x)
      x = F.relu(x)
      x = self.l6(x)
      x = self.output(x)

      return x

    def train_loop(self, trainloader, device, loss_fn, optimizer, batch_size=None) -> None:
        for batch in tqdm(trainloader):
            image, label = batch[0], batch[1]

            # GPU に移行
            image = image.to(device)
            label = label.to(device)

            # 勾配を0で初期化（これをしないと逆伝播で計算した勾配が足されていく）
            optimizer.zero_grad()

            # 順伝播
            pred = self.forward(image)

            # ロスの計算
            loss = loss_fn(pred, label)

            # 逆伝播
            loss.backward()

            # パラメータの更新
            optimizer.step()


    def test_loop(self, testloader, device) -> list:
        with torch.no_grad():
            acc_list = []
            for batch in tqdm(testloader):
                image, label = batch[0], batch[1]
                image = image.to(device)
                label = label.to(device)
                pred = self.forward(image)
                accuracy = 100 * torch.sum(torch.argmax(pred, dim=1) == label) / len(pred)
                acc_list.append(float(accuracy))
            print(f'Accuracy {np.mean(acc_list):.2f}')

        return acc_list
