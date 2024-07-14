# NNGPUBenchmark
MLPやCNNでGPUの性能をベンチマークする

##  環境
* python 3.10
* cuda 11.8

## 使い方
ニューラルネットワークモデルを動かすだけなら以下のコマンドをたたく．
```bash
python main.py
```

MLPでベンチマークをしたいときは以下のコマンド．
```bash
python benchmark.py MLP
```

CNNでベンチマークをしたいときは以下のコマンド．
```bash
python benchmark.py CNN
```
