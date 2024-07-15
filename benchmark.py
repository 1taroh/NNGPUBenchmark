import sys
import torch
import datetime
import time
from model import MLP, CNN
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms

def main():
    model_name = get_model_name()
    print(f"The selected model is: {model_name}")

    train_batch_size_list = [64, 128, 256, 512, 1024]
    train_batch_size_list = [64, 128, 256, 512, 1024]

    now = datetime.datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S_benchmark_results.txt")
    filepath = 'results/'+filename

    print('Start Benchmark')
    for train_batch_size in train_batch_size_list:
        for test_batch_size in train_batch_size_list:
            benchmark_results = benchmark(model_name, train_batch_size, test_batch_size)
            #TODO: 様々なbatch_size でベンチマークしたい
            save_benchmark_results(model_name, benchmark_results, filepath)
            print('Benchmark results are saved.')
    print('End Benchmark')

    print('All benchmark results are saved.')

def benchmark(model_name, train_batch_size, test_batch_size) -> list:
    epochs = 10

    if model_name == 'MLP':
        benchmark_results = benchmark_MLP(epochs, train_batch_size, test_batch_size)
    elif model_name == 'CNN':
        benchmark_results = benchmark_CNN(epochs, train_batch_size, test_batch_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Expected 'MLP' or 'CNN'.")
        
    return benchmark_results

def benchmark_MLP(epochs=3, train_batch_size=128, test_batch_size=256) -> list:
    start = time.time()

    ####### Start Benchmark ########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MLP()
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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    acc_list = []
    # モデルの学習
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        model.train_loop(trainloader, device, loss_fn, optimizer)

        acc = model.test_loop(testloader, device)
        acc_list.extend(acc)
    ####### End Benchmark ########

    end = time.time()
    training_time = end - start
    accuracy = max(acc_list)

    benchmark_results = [
        training_time,
        epochs, 
        train_batch_size,
        test_batch_size,
        accuracy
        ]
    return benchmark_results

def benchmark_CNN(epochs=3, train_batch_size=128, test_batch_size=256) -> list:
    start = time.time()

    ####### Start Benchmark ########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    acc_list = []
    # モデルの学習
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        model.train_loop(trainloader, device, loss_fn, optimizer)

        acc = model.test_loop(testloader, device)
        acc_list.extend(acc)
    ####### End Benchmark ########

    end = time.time()
    training_time = end - start
    accuracy = max(acc_list)

    benchmark_results = [
        training_time,
        epochs, 
        train_batch_size,
        test_batch_size,
        accuracy
        ]
    return benchmark_results

def get_model_name() -> str:
    # Check if an argument was provided
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        # If no argument was provided, ask the user for input
        model_name = input("Please enter a model name: ")
    return model_name

def save_benchmark_results(model_name, benchmark_results, filepath) -> None:
    training_time    = benchmark_results[0]
    epochs           = benchmark_results[1]
    train_batch_size = benchmark_results[2]
    test_batch_size  = benchmark_results[3]
    accuracy         = benchmark_results[4]

    with open(filepath, 'a') as f:
        f.write(torch.cuda.get_device_name())
        f.write('\n')
        f.write(f'model name: {model_name}\n')
        f.write(f'Training time: {training_time:.2f}[sec]\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Train Batch Size: {train_batch_size}\n')
        f.write(f'Test Batch Size: {test_batch_size}\n')
        f.write(f'Accuracy: {accuracy:.2f}%\n')

if __name__ == "__main__":
    main()
