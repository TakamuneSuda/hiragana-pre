# 畳み込みニューラルネットワークを用いた手書きひらがなの予測

畳み込みニューラルネットワークCNNを用い、手書きひらがなの予測をします。

# ファイル内容  
* classmapping.csv  
    新たなラベル・JISコード・ひらがなを内包
* Preprocess.ipynb  
    手書きひらがなデータベースETL8Gを学習可能なNamPy配列に前処理する。  
* Model.py  
   CNNの定義を記載
* train.ipynb  
    Model.pyで設定したCNNモデルとPreprocess.ipynbで作成したNamPy配列'hira.npz'から訓練を行う。
    
    
*
