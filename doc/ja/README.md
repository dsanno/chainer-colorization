# Chainer implementation of Colorization

# 参考にした論文
[ディープネットワークを用いた大域特徴と局所特徴の学習による 白黒写真の自動色付け](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/ja/)

# 論文との違い

* 画像の特徴抽出に[VGG 16 layers model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)を使用。
* Classificationの学習を行っていない。


# 必要な環境

* Python 2.7
* [Chainer 1.8.0](http://chainer.org/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)

# 使用方法

## 学習データとなる画像ファイルの配置

学習データとなる画像ファイルを1つのディレクトリの中にすべてコピーしてください。

## VGG 16 layers モデルのダウンロード
[ここ](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)からVGG_ILSVRC_16_layers.caffemodelをダウンロードして、ルートディレクトリに置いてください。

## VGG 16 layers モデルを Chainer で扱えるように変換

ChainerでCaffe modelを読み込むのに時間がかかるので、Chainerで読み込みやすい形式に変換かつ必要のないパラメータを削除します。

```
$ python src/create_chainer_model.py
```

## 学習データの変換

画像ファイル群をpklファイルに変換します。

```
$ python src/convert_dataset.py <画像ファイルのあるディレクトリ> <出力ファイル名> -n <最大ファイル数>
```

例:

```
$ python src/convert_dataset.py dataset/image dataset/images.pkl -n 200000
```

## モデルの学習

例:

```
$ python src/train.py -g 0 -m vgg16.model -o model/color --out_image_dir image -d dataset/images.pkl --batch_size 48
```

オプション

* -g (--gpu) <GPUデバイス番号>: 任意  
GPUデバイス番号を指定します。負の値を指定した場合はCPUを使用します(デフォルト: -1)
* -m (--model) <VGG 16 layers モデルファイルパス>: 任意  
変換後のVGG 16 layersモデルのファイルパスを指定してください(デフォルト: vgg16.model)
* -i (--input) <モデルファイルパス>: 任意  
入力する学習モデルのファイルのパスを拡張子を除いて指定してください
* -o (--output) <モデルファイルパス>: 必須  
出力する学習モデルのファイルのパスを拡張子を除いて指定してください  
保存されるモデルファイル名以下の通りです
    * モデルパラメータファイル: <モデルファイルパス>_<イテレーション番号>.model
    * optimizerパラメータファイル: <モデルファイルパス>_<イテレーション番号>.state
* -d (--dataset) <データセットファイルパス>: 任意  
データセットファイルパスを指定します(デフォルト: dataset/images.pkl)
* --iter <イテレーション数>: 任意  
イテレーション数を指定します(デフォルト: 100)
* --batch_size <ミニバッチ数>: 任意  
ミニバッチ数を指定します(デフォルト: 48)
* --out_image_dir <画像出力ディレクトリ>: 任意  
学習途中に学習データを使って変換した画像を出力します  
指定しない場合は画像の出力を行いません

# ライセンス

MIT
