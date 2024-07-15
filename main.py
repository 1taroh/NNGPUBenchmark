from model import MLP, CNN
import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = MLP()
model = CNN()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# データの前処理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 のダウンロード、読み込み
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8192, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=8192, shuffle=False)

epochs = 10
acc_list = []
# モデルの学習
for epoch in range(epochs):
    print(f'Epoch {epoch}:')
    model.train_loop(trainloader, device, loss_fn, optimizer)

    acc = model.test_loop(testloader, device)
    acc_list.extend(acc)

end = time.time()

print(f"time: {end-start:.2f}[sec]")
plt.plot(acc_list)
plt.show()
