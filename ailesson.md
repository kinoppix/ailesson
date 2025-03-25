# Python & PyTorch を使った AI プログラミング研修

---

## はじめに：環境セットアップについて

本教材では **Python 3** と **PyTorch** を使用します。  
実行環境
- WindowsのVisual Studio Code(以下、VS Code)を利用
- コンピューター室の機械学習PCにVS Codeからリモート接続

機械学習PCは以下の環境がインストールされています

- **Python**
- **PyTorch**


【上級者向け】もっと学びたい人はCursorの使用をおすすめします。実行結果を丁寧に解説してくれるので理解が深まります。

---

# 第1部：ニューラルネットワーク(NN)の基礎

## 1.1 機械学習・ディープラーニングの基本概念

### 1.1.1 機械学習とは
学習とは、 **後から与えられたデータをもとに動作が変わる事象** のこと。機械でこの事象を実現しているので **機械学習** という。

- **機械学習**: コンピュータが大量のデータをもとにパターンを学習し、未知のデータに対して推測を行う技術。
  - **教師あり学習**: 正解ラベルがあるデータを使って学習。例えば画像分類
  - **教師なし学習**: 正解ラベルがないデータの構造を学習。例えば歩き方の学習
  - **強化学習**: 報酬を最大化するように学習。NNだけでなく遺伝的アルゴリズム(GA)でも用いられる。

強化学習の例
[【物理エンジン】人工知能にフライパン返しを学習させたら職人になったｗ](https://www.youtube.com/watch?v=DDhv7biBd4o)

### 1.1.2 ディープラーニング（深層学習）とは
- **ディープラーニング**: 多層のニューラルネットワークを用いて複雑なパターンを学習する機械学習の一分野。  
- 特徴抽出から予測まで、すべてを一貫して学習できる点が大きな強み。

古典的なニューラルネットワーク
![alt text](https://cvml-expertguide.net/wp-content/uploads/2022/07/58307497beea0f403f30b517b7700b70-768x742.png)

ディープラーニング
![alt text](https://cvml-expertguide.net/wp-content/uploads/2021/08/CNN-1.png)

---

## 1.2 ニューラルネットワークの仕組み

### 1.2.1 ニューロン
- **ニューロン（パーセプトロン）**: 生物の神経細胞を模したモデル。  
  入力 \( x \) に対し重み \( w \) とバイアス \( b \) を用いて線形変換 \( w \cdot x + b \) を行い、活性化関数で非線形を与える。

### 1.2.2 活性化関数
- ニューラルネットワークに非線形性を与える関数。代表的なもの:  
  - **シグモイド関数 (sigmoid)**
  - **ReLU**  
  - **tanh**

  例：  
  \[
  \text{ReLU}(x) = \max(0, x)
  \]

### 1.2.3 誤差逆伝播（バックプロパゲーション）
- **誤差逆伝播**: NN が出力した結果と正解の誤差を逆方向に伝播させて、各重みを更新する仕組み。  
  PyTorch では自動で勾配を計算してくれる（**autograd** 機能）。

---

## 1.3 PyTorchの導入（テンソル操作・計算グラフ・勾配と最適化）

### 1.3.1 テンソル(Tensor)とは
- **テンソル**: 多次元配列を扱うためのデータ構造。NumPy の ndarray に近いが、GPU での計算が可能。

```python
# サンプルコード：PyTorchテンソルの作成と基本操作
import torch

# 1次元のテンソル（ベクトル）
x = torch.tensor([1.0, 2.0, 3.0])
print("x:", x)

# 2次元のテンソル（行列）
y = torch.tensor([[1.0, 2.0], 
                  [3.0, 4.0]])
print("y:", y)

# テンソル同士の加算
z = x + 10  # ブロードキャストがかかる例
print("z:", z)
```

### 1.3.2 計算グラフと自動微分 (autograd)
- PyTorch のテンソルを `requires_grad=True` に設定すると、計算の過程がグラフとして記録され、自動的に微分を求められる。

```python
# サンプルコード：自動微分
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)
b = a * 2  # ここで計算グラフが作成される
loss = b.sum()  # スカラーを作る

loss.backward()  # 誤差逆伝播をシミュレート
print(a.grad)    # aに対応する勾配が自動計算
```

### 1.3.3 最適化 (optimizer)
- PyTorch は重みの更新（学習率、オプティマイザの種類など）を自動的に処理する機能がある。
- 代表的な最適化手法: **確率的勾配降下法 (SGD)**、**Adam** など。

---

## 1.4 PyTorch で簡単な NN を作成する演習

ここでは、**線形回帰**をシンプルな例題として取り上げます。  
高次元のデータや分類問題でも考え方は同じです。

### 1.4.1 線形回帰の基本
- **線形回帰**: ある入力 \( x \) に対して、 \( y = w \cdot x + b \) という直線で近似するモデルを学習する。

### 1.4.2 演習コード例

```python
"""
【Google Colab の場合は以下を実行して PyTorch をインストール】
!pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1) データセットの作成
# y = 2 * x + 1 を元にノイズを加えたデータを用意
# ここではランダムに100個のデータを作る
true_w = 2.0
true_b = 1.0
x_train = torch.linspace(-1, 1, 100).unsqueeze(1)  # shape: (100, 1)
y_train = true_w * x_train + true_b + 0.3 * torch.randn(x_train.size())

# 2) モデルの定義（単純な全結合層を1つ）
model = nn.Linear(in_features=1, out_features=1)

# 3) 損失関数と最適化手法の設定
criterion = nn.MSELoss()           # 平均二乗誤差
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 学習率0.1のSGD

# 4) 学習ループ
num_epochs = 100
loss_list = []

for epoch in range(num_epochs):
    # 順伝播：予測値を計算
    y_pred = model(x_train)
    
    # 損失を計算
    loss = criterion(y_pred, y_train)
    
    # 逆伝播：勾配を初期化してから計算し、重みを更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss.item())

    # 10エポックごとに経過を表示
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 5) 結果の可視化
plt.figure(figsize=(8,4))
# 学習データの散布図
plt.scatter(x_train.detach().numpy(), y_train.detach().numpy(), label='Train Data')
# モデルの予測直線
x_plot = torch.linspace(-1, 1, 100).unsqueeze(1)
y_plot = model(x_plot)
plt.plot(x_plot.detach().numpy(), y_plot.detach().numpy(), color='red', label='Fitted Line')
plt.legend()
plt.show()

# 学習されたパラメータを表示
learned_w = model.weight.item()
learned_b = model.bias.item()
print(f"Learned w = {learned_w:.2f}, Learned b = {learned_b:.2f}")
```

#### コードのポイント
1. **データの作成**: `torch.linspace` などを用いて連続的な入力データを作成。ノイズを加えて実際っぽさを演出。  
2. **モデルの定義**: `nn.Linear` は線形変換（ \( y = w \times x + b \) ）を実装。  
3. **損失関数**: `nn.MSELoss()` は平均二乗誤差を計算。  
4. **最適化**: `optim.SGD` を使い、勾配をもとにパラメータを更新。  
5. **学習ループ**:  
   - `optimizer.zero_grad()` で勾配をリセット  
   - `loss.backward()` で勾配計算  
   - `optimizer.step()` で重み更新  
6. **可視化**: 学習データの散布図と学習済みモデルの直線をプロット。  
7. **結果の確認**: 最終的に学習した `w` と `b` が、初期設定した `true_w=2.0`、`true_b=1.0` に近い値になっているかチェック。

---

## 1.5 まとめ
- ニューラルネットワークは、入力から出力を導く変換（重み + 活性化関数）を多層に重ねたモデル。  
- バックプロパゲーション（誤差逆伝播）により、誤差を最小化するように重みを更新する仕組み。  
- PyTorch では `nn.Module` を継承したクラスや `nn.Linear` などを使い、少ないコード量で NN を構築できる。  
- `autograd` 機能により、手動で微分計算をする必要がない。

---

## 1.6 確認問題

1. **ニューラルネットワーク** は何をモデル化したものでしょうか？  
2. **活性化関数** の役割は何でしょうか？  
3. PyTorch における **自動微分 (autograd)** の仕組みを簡単に説明してください。  
4. 簡単な線形回帰モデルを学習する際、**損失関数** として何を用いるのが一般的ですか？  
5. PyTorch でモデルを学習するとき、パラメータの更新処理に必要な関数を 2 つ挙げてください。

---

# 第2部：MNIST データセットを使った画像認識演習

## 2.1 CNN（畳み込みニューラルネット）の直感的説明

### 2.1.1 畳み込み（Convolution）
- 画像にフィルタ（カーネル）をかけて特徴を抽出する操作。周辺ピクセル情報を考慮しやすい。  
- 全結合層に比べてパラメータ数を大幅に削減でき、画像認識に強みがある。

### 2.1.2 プーリング（Pooling）
- 畳み込み層で得られた特徴マップを縮小する操作。主に **最大値プーリング (Max Pooling)** などを使用。

### 2.1.3 CNN の全体構造
- **畳み込み層** → **プーリング層** → **全結合層** (分類部) の流れ。  
- 入力画像を特徴マップに変換 → クラスに分類する。

---

## 2.2 MNIST データセットの取得と可視化

### 2.2.1 MNIST とは
- 手書き数字 (0～9) の 28×28 ピクセルの白黒画像が 6 万枚（学習用）+ 1 万枚（テスト用）あるデータセット。  
- PyTorch では `torchvision.datasets` を用いて簡単に取得可能。

### 2.2.2 データ取得サンプルコード

```python
"""
【Google Colab の場合は以下を実行して PyTorch をインストール】
!pip install torch torchvision
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# MNISTのデータをダウンロード＆読み込み準備
# transforms.ToTensor() で画像を Tensor 化して [0,1] の範囲に正規化
train_dataset = torchvision.datasets.MNIST(
    root='./data',         # データの保存先
    train=True,            # 学習用データを取得する
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

print("学習データ数:", len(train_dataset))
print("テストデータ数:", len(test_dataset))

# 一部の画像を可視化
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

---

## 2.3 PyTorch による CNN モデルの作成、学習および評価

### 2.3.1 CNN モデルの定義例

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 畳み込み層1: 入力チャネル=1, 出力チャネル=16, カーネルサイズ=3
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 畳み込み層2: 入力チャネル=16, 出力チャネル=32, カーネルサイズ=3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 全結合層: 32 * 7 * 7 -> 10 (MNISTは最終クラス10)
        self.fc = nn.Linear(32*7*7, 10)
        
    def forward(self, x):
        # 畳み込み1 -> ReLU -> プーリング
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # 畳み込み2 -> ReLU -> プーリング
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # テンソルを平坦化 (batch_size, 32*7*7)
        x = x.view(x.size(0), -1)
        
        # 全結合層 -> 出力は (batch_size, 10)
        x = self.fc(x)
        return x
```

### 2.3.2 データローダーの準備
- **DataLoader** を使うと、小分け（ミニバッチ）にしてデータを読み込める。

```python
from torch.utils.data import DataLoader

batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

### 2.3.3 学習と評価

```python
# モデルのインスタンス化
model = SimpleCNN()

# デバイス選択 (GPUが使える場合は利用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 損失関数、最適化関数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 勾配初期化
        optimizer.zero_grad()
        # 順伝播
        outputs = model(images)
        # 損失
        loss = criterion(outputs, labels)
        # 逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        
        running_loss += loss.item()
    
    # エポックごとの損失平均を表示
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# テストデータでの評価
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

## 2.4 ハイパーパラメータ調整を含む実践的演習

- **ハイパーパラメータ**: 学習率 (lr), バッチサイズ (batch_size), エポック数 (num_epochs), 畳み込み層のフィルタ数など。
- 練習として下記を変えてみる:
  - 学習率: 0.001 → 0.01 → 0.1
  - バッチサイズ: 32 → 64 → 128
  - 畳み込み層のチャネル数: (16, 32) → (32, 64)
  - エポック数を増やす：5 → 10  
- 変更すると **学習速度** や **精度** がどのように変化するか観察する。

### 演習課題例
1. 学習率を 0.01 に変えて実験し、Loss の減り方や Test Accuracy がどう変わるかを調べる。  
2. バッチサイズを 128 に変えた場合も同様に実験。  
3. 畳み込み層のチャネル数を増やす（例: conv1 を 32、conv2 を 64 に）してみる。

---

## 2.5 まとめ
- CNN は畳み込み層とプーリング層を用いて画像特徴を抽出し、全結合層で分類を行うモデル。  
- MNIST は手書き数字認識に特化した代表的データセットで、初学者向け実習素材として最適。  
- **DataLoader** を活用することで、効率的にバッチ学習が可能。  
- ハイパーパラメータ（学習率、バッチサイズ、チャネル数など）を調整することでモデルの性能が大きく変わる。

---

## 2.6 確認問題

1. **CNN(畳み込みニューラルネットワーク)** が、画像認識でよく使われる理由は何でしょうか？  
2. **畳み込み (Convolution)** と **プーリング (Pooling)** の役割を簡単に説明してください。  
3. MNIST の画像サイズは何×何ピクセルで、何種類のクラス分類問題でしょうか？  
4. ハイパーパラメータ調整で結果が大きく変わる例を 1 つ挙げ、その理由を簡単に説明してください。  
5. PyTorch で訓練時と推論時で挙動を変える場合、必ず呼び出すべきメソッドを答えてください。  

---

# 第3部：生成AI入門（学習から生成までの応用）

## 3.1 生成AIの基本概念
- **生成AI**: 新しいデータ（画像や文章など）を生成するモデル。  
  - 画像生成では **GAN** (Generative Adversarial Network) や **拡散モデル (Diffusion Model)**  
  - 文章生成では **Transformer** ベースのモデル (GPT など)

### 3.1.1 GAN の基本アイデア
- **ジェネレータ (Generator)** と **ディスクリミネータ (Discriminator)** の 2 つのネットワークを対戦 (Adversarial) させることで、リアルなデータに近い生成を目指す。

### 3.1.2 拡散モデル (Diffusion Model) の基本アイデア
- 画像にノイズを徐々に加えていく過程（拡散）と、そこから元の画像を復元する過程（生成）を学習。  
- 最近の高品質画像生成で注目を浴びている。

### 3.1.3 Transformer の基本アイデア
- Attention 機構を用いて、時系列上の文脈を効率よく捉える。  
- GPT 系や BERT 系モデルが有名で、自然言語処理に強み。

---

## 3.2 PyTorch を用いた簡単な画像生成モデル構築

ここでは、GAN の入門例として **DCGAN (Deep Convolutional GAN)** を非常に簡略化した形で紹介します。  
※ 実際にはより多くのハイパーパラメータ調整や層の設計が必要ですが、学習経過を体験するための小規模サンプルです。

### 3.2.1 データセットの用意 (MNIST を利用)
GAN で学習するため、MNIST の手書き数字画像を使って“数字のように見える”画像を生成することを目標にします。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# すでに MNIST の train_dataset を用意している前提
# 生成モデル用にラベル情報は不要、イメージだけ使用

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 3.2.2 ジェネレータとディスクリミネータの定義

```python
# ジェネレータ: 潜在変数(乱数)から28x28の画像を生成
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # 全結合層で7x7x128程度の特徴マップを作る
            nn.Linear(latent_dim, 128*7*7),
            nn.ReLU(True),
            
            # Reshape操作はviewで行う
            # 転置畳み込みで画像サイズを28x28に拡大
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 出力を[-1,1]に
        )
    def forward(self, z):
        return self.main(z)

# ディスクリミネータ: 入力画像が本物か偽物かを二値分類
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()  # 本物(1)、偽物(0)の確率を出す
        )
    def forward(self, img):
        return self.main(img)
```

### 3.2.3 学習ループ (簡略版)

```python
latent_dim = 100
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

num_epochs = 5

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        
        # ---------------------
        # 1. Discriminatorの学習
        # ---------------------
        # 本物画像
        real_imgs = imgs.to(device)
        real_validity = D(real_imgs)
        
        # 本物ラベルは1
        real_labels = torch.ones(real_imgs.size(0), 1, device=device)
        
        # 偽画像
        z = torch.randn(real_imgs.size(0), latent_dim, device=device)
        fake_imgs = G(z)
        fake_validity = D(fake_imgs.detach())
        
        # 偽ラベルは0
        fake_labels = torch.zeros(real_imgs.size(0), 1, device=device)
        
        # 本物損失 + 偽損失
        loss_real = criterion(real_validity, real_labels)
        loss_fake = criterion(fake_validity, fake_labels)
        loss_D = (loss_real + loss_fake) / 2
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        
        # -----------------
        # 2. Generatorの学習
        # -----------------
        # Generatorが作った偽物をDiscriminatorに「本物」と判定させたい
        fake_validity = D(fake_imgs)
        loss_G = criterion(fake_validity, real_labels)  # 本物ラベル=1を期待
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
        if i % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(train_loader)} "
                  f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")
    
    # エポックごとにサンプル画像を保存
    z = torch.randn(64, latent_dim, device=device)
    generated_imgs = G(z)
    save_image(generated_imgs[:25], f"epoch_{epoch+1}.png", nrow=5, normalize=True)
```

- 学習を続けると、最初はノイズにしか見えなかった画像が、徐々に数字らしくなるのを確認できる。

---

## 3.3 自然言語生成（簡単な Transformer ベースの文章生成モデル）の基礎演習
GAN と同様、ここでは簡易的なミニ Transformer を用いたサンプルを示します。  
※ 実際には巨大なモデル（GPT など）を学習するのは計算リソース面で非常に大変なので、原理理解用のコード例にとどめます。

### 3.3.1 単純な文章生成の流れ
1. テキストデータの **トークナイズ**（単語やサブワードに分割）  
2. Transformer モデルで次の単語を予測するよう学習  
3. 学習したモデルを使って、スタートトークンから文章を生成  

### 3.3.2 PyTorch の `nn.Transformer` 簡易使用例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer

# サンプル：極めて小規模な Transformer で文字列予測をやってみる
# データ例: 文字列 "hello" を学習して次の文字を予測

# 文字辞書を作成（簡易例）
chars = list("abcdefghijklmnopqrstuvwxyz ")
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

def encode_text(text):
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)

def decode_text(indices):
    return "".join(idx2char[i.item()] for i in indices)

# 学習用のペア（入力= "hell", 出力= "ello"）のように作る
input_text = "hell"
target_text = "ello"

input_ids = encode_text(input_text).unsqueeze(1)  # shape: (length, batch)
target_ids = encode_text(target_text).unsqueeze(1)

vocab_size = len(chars)
embed_size = 32
num_heads = 2
hidden_dim = 64
num_layers = 2

# 簡易的に Transformer を使う
transformer_model = Transformer(
    d_model=embed_size,
    nhead=num_heads,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers,
    dim_feedforward=hidden_dim
)

# エンベディング層と出力層
src_embedding = nn.Embedding(vocab_size, embed_size)
tgt_embedding = nn.Embedding(vocab_size, embed_size)
out_linear = nn.Linear(embed_size, vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(transformer_model.parameters()) + 
                       list(src_embedding.parameters()) +
                       list(tgt_embedding.parameters()) + 
                       list(out_linear.parameters()), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    
    src_emb = src_embedding(input_ids)  # shape: (length, batch, embed_size)
    tgt_emb = tgt_embedding(target_ids[:-1])  # decoder入力は一つ前まで
    
    # Transformerは (seq_len, batch, embed_dim) で処理する想定
    # src_emb, tgt_emb ともに (length, batch, embed_size) の形
    
    output = transformer_model(src_emb, tgt_emb)  # shape: (tgt_len, batch, embed_size)
    
    logits = out_linear(output)  # (tgt_len, batch, vocab_size)
    
    # 正解は target_ids[1:] (最初の文字を除いた部分)
    loss = criterion(logits.reshape(-1, vocab_size), target_ids[1:].reshape(-1))
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 学習後、生成テスト: "hell"から次の文字を推定
src_emb = src_embedding(input_ids)
tgt_emb = tgt_embedding(target_ids[:-1])
output = transformer_model(src_emb, tgt_emb)
logits = out_linear(output)
pred_id = torch.argmax(logits[-1,0])  # 最後のタイムステップの出力
print("Predicted next char:", idx2char[pred_id.item()])
```

- 上記の例は極端に単純化しており、実用性はありませんが、Transformer を用いて「次の文字を予測する」流れを体感できます。  
- 実際の大規模言語モデル（GPT など）では、膨大なデータとモデルサイズを用いて学習します。

---

## 3.4 作成したモデルを使った生成結果の評価と分析のポイント
1. **主観評価**: 生成された画像や文章を人間が見て判断する。  
2. **定量的評価**: 画像なら FID (Fréchet Inception Distance)、文章なら BLEU スコアなどの指標で評価する。  
3. **多様性** と **品質** のバランス: 同じクラスでも多様なパターンを生成できるか。  
4. **過学習のチェック**: 学習データをそのまま丸暗記していないか（同じ画像を再現していないか）。

---

## 3.5 まとめ
- **生成AI** は新しいデータ（画像や文章）を作り出すモデルであり、GAN や拡散モデル、Transformer など多彩な手法がある。  
- GAN では Generator と Discriminator を競わせ、よりリアルな生成を学習する仕組み。  
- 拡散モデルはノイズ付与と復元のプロセスを学習し、高品質な画像生成を可能にする。  
- Transformer は Attention 機構を用いて時系列データ（特に言語）を効率的に扱う。  
- 生成した結果の評価には定量的/定性的なアプローチを組み合わせることが大切。

---

## 3.6 確認問題

1. **GAN のジェネレータ** と **ディスクリミネータ** は、それぞれどのような役割を持っていますか？  
2. 拡散モデルが画像生成を行う際、どのような 2 つのプロセスを学習しますか？  
3. **Transformer** で用いられる **Attention** 機構の利点は何でしょうか？  
4. 生成モデルの **評価指標** として、画像・文章それぞれで使われる例を挙げてください。  
5. 生成モデルの問題点の 1 つとして挙げられる **過学習やモラル・倫理的問題** について、どのようなことに注意すべきでしょうか？

---

# 全体を通しての学習の流れ・ポイント

1. **Python と PyTorch の基本** を押さえ、テンソル操作や NN 構築の流れを理解する（第1部）。  
2. **CNN を用いた画像認識** で、ディープラーニングの具体的な学習・評価を体験する（第2部）。  
3. **生成AI の初歩**（GAN や拡散モデル、Transformer など）を実装し、新たなデータを“生み出す”楽しさと難しさを実感する（第3部）。

各部の内容をしっかりと理解し、確認問題や演習課題を行ってみてください。  
わからない部分は公式ドキュメントや書籍、インターネット情報を参考にするとよいでしょう。  
今後さらに応用として、自然言語処理や強化学習、ほかの生成モデルや大規模言語モデルの学習など、さまざまな方向へ発展できます。

---

## 最後に
- **まとめノート**: 学んだ内容を自分の言葉で整理し、疑問点や新しく知ったキーワードなどを書き留めておきましょう。  
- **実行して試す習慣**: コードを自分で実行・修正して挙動を観察することが、理解を深める近道です。  
- **他の応用例やデータセットにもチャレンジ**: CIFAR-10 や自然言語データなどを扱うと、より汎用的なスキルを磨けます。

本教材が、高校生の皆さんがディープラーニングと PyTorch の楽しさを体感し、AI の可能性を広げる一助となれば幸いです。
