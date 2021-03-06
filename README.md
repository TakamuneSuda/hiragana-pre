# 畳み込みニューラルネットワークを用いた手書きひらがなの予測

## このコードについて  
畳み込みニューラルネットワークCNNを用い、手書きひらがなの予測をします。精度は99.30%です。

## ファイル内容  
* [classmapping.csv](https://github.com/TakamuneSuda/hiragana-pre/blob/master/classmapping.csv)  
    新たなラベル・JISコード・ひらがなを内包
* [Preprocess.ipynb](https://github.com/TakamuneSuda/hiragana-pre/blob/master/Preprocess.ipynb)  
    手書きひらがなデータベースETL8Gを学習可能なNamPy配列に前処理する。  
* [Model.py](https://github.com/TakamuneSuda/hiragana-pre/blob/master/Model.py)  
   CNNの定義を記載。
* [Train.ipynb](https://github.com/TakamuneSuda/hiragana-pre/blob/master/Train.ipynb)  
    Model.pyで設定したCNNモデルとPreprocess.ipynbで作成したNamPy配列'hira.npz'から訓練を行う。
* [Prediction.ipynb](https://github.com/TakamuneSuda/hiragana-pre/blob/master/Predeiction.ipynb)  
    train.ipynbで構築したモデルから、実際に手書きひらがなの予測を行う。予測する画像は'pre_data'に入っている。
* [cnn.py](https://github.com/TakamuneSuda/hiragana-pre/blob/master/cnn.py)  
    modeをコマンドライン引数として指定することで、訓練・予測ができるプログラム。
  
## 環境  
* Python 3.6.4 (Anaconda)
* TensorFlow 1.12.0

## 流れ  
1. データの準備  
    産業技術総合研究所が配布している手書教育漢字データベースＥＴＬ８をダウンロード。  
    http://etlcdb.db.aist.go.jp/?page_id=2461&lang=ja  
    (この中には、のべ1600人分の教育漢字881文字, ひらがな75文字が納められている。)  
1. 前処理  
    'Preprocess.ipynb'を使い、ダウンロードしたデータベースからNamPy配列'hira.npz'を作る。  
1. モデル設定  
    訓練させるCNNモデルを構築する。関数で定義する為に、'Model.py'で保存。  
1. 訓練  
    'Model.py'で設定したモデルと前処理を行った'hira.npz'から実際に訓練を行う。予測ができるようにmodelディレクトリに保存する。  
1. 予測  
    訓練済みモデルから実際に手書きひらがなの予測を行う.
    
## 追記  
コマンドライン引数を指定し訓練・予測できるプログラムを追加しました。  
#### 訓練モード  
コマンドライン引数のmodeにtrainを入れると訓練が行われる。  
同ディレクトリ内にある'hira.npz'を使用し、60ループ×500step=300000回の学習が行われ、モデルは'etl8g_convnet_model'に保存される。  
&nbsp;　`python3 cnn.py --mode=train`  

#### 予測モード  
コマンドライン引数のmodeにpredictを入れると予測が行われる。  
image_fileに予測させる画像を指定することで、訓練モードで学習したモデルから予測させたひらがなを出力する。  
&nbsp;　`python3 cnn.py --mode=predict --image_file=./pre_data/1.png`  
&nbsp;    　　　   <img src="https://i.imgur.com/Lx1QYVR.png" width="200px">  
&nbsp; 　　　　　　./pre_data/1.png  
&nbsp; 出力  
&nbsp;　`画像のひらがなは「い」です`
    
