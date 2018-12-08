# 畳み込みニューラルネットワークを用いた手書きひらがなの予測

## このコードについて  
畳み込みニューラルネットワークCNNを用い、手書きひらがなの予測をします。精度は98.72%です。

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
