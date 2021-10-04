# Environment
OS Ubuntu 18.04.5
PyTorch 1.8.1+cu102
Python 3.6.9
RTX2080

# Usage
## Natural training
python3 train.py --training_mode natural

## Adversarial training
python3 train.py --training_mode adversarial

## Natural test
python3 test.py --test_mode natural

## Adversarial attack
python3 test.py --test_mode adversarial

# ディレクトリ構成
## utils.py
主に実験の記録用
- 再実験用に実験時のパラメータをスクリプトとして保存，モデルを.pt形式で保存

## model.py
実験タスクごとにネットワークを定義

### Architecture Searchの実装部分
Architecture Searchは主に以下で構成されている．
- train_search.py
- model_search.py
- architect.py

--- train_search.py ---
```python
from model_search import Network
...
# args.init_channels : 16
# CIFAR_CLASSES : 10
# args.layers : 8
# criterion : nn.CrossEntropyLoss()
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
```

--- model_search.py ---
```python
class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        ...
        self.cells = nn.ModuleList()
        ...
        for i in range(layers):
            ...
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells += [cell]
```

```python
class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):

```


### ネットワーク(モデル)の実装部分
train.pyでmodelを呼び出している．  
NetworkCIFARでは層の深さ分，つまりlayers個のcellを作成し，ModuleList型のリストcellsにappendしている．  
Cellはgenotypeから演算のリストを取り出し，

--- train.py ---  
```python
from model import NetworkCIFAR as Network
...
# args.init_channels = 36
# CIFAR_CLASSES = 10
# args.layers = 20
# args.auxiliary = False
# genotype = eval("genotypes.%s" % args.arch)
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
```

--- model.py ---  

```python
class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
    ...
    self.cells = nn.ModuleList()  
    ...
    for i in range(layers):
        ...  
        cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
        ...
        self.cells += [cell]
```

```python
class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        ...
        ## op_name: 演算名が書かれた文字列のリスト
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        ...
        self._ops = nn.ModuleList()

        # OPS : 演算名をキー，データがlambda式の辞書型データ
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices
```

--- genotypes.py ---

```python
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_V2 = Genotype(
    normal = [
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat = [2, 3, 4, 5],   # concat : 連結
    reduce = [
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2), 
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

DARTS = DARTS_V2
```

--- operations.py ---
```python
import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}
```

## genotype.py
アーキテクチャの構成をnamedtupleで実装している．
namedtupleは辞書型っぽくアクセスできるタプルでタプルと同様でimmutableである．使い方は以下参照．

```python
from collections import namedtuple
Car = namedtuple('Car' , 'color mileage')
my_car = Car('red', 3812.4)
my_car.color
'red'
my_car.mileage
3812.4
```

--- train.py---
genotype = eval("genotypes.%s" % args.arch)
model = Network(..., genotype)

--- model.py ---
class NetworkCIFAR(nn.Module):
    def __init__(..., genotype):


## operations.py
新しい演算を定義．以下の6種類を新たに定義
- ReLUConvBN
- DilConv ('dil_conv_3x3', 'dil_conv_5x5')
- SepConv ('sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7')
- Identity ('skip_connect')
- Zero
- FactorizedReduce

### ReLUConvBN


### DilConv


### SepConv


### Identity
恒等写像: f(x)=x

### Zero


### FactorizedReduce
[paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w10/Wang_Factorized_Convolutional_Neural_ICCV_2017_paper.pdf)