# M5 Forecasting - Accuracy
#### Estimate the unit sales of Walmart retail goods

## 概要  
(Public) stage1では1914日目から1941日目までの予測を行う。
(Private) stage2(コンペの最終評価)では1942日目から1969日目までの予測を行う。    
ひとまずは与えられたcsvデータを用いて1914-1941日を予測する  

## data
### calendar.csv
2011/01/29 ~ 2016/06/19 までの1969日間のカレンダー情報  

### sales_train_validation.csv  
1913日間(訓練データ)の売り上げ情報  
train: d_1(2011/01/29)  ~ d_1913(2016/04/24)

### sample_submission.csv
ある匿名のアイテム30490個が未来の28日間で何個売れるかを予測する  
stage1: d_1914(2016/04/25) ~ d_1941(2016/05/22)  コンペ終了1ヶ月前(6/1)に正解データが与えられる
stage2: d_1942(2016/05/23) ~ d_1969(2016/06/19)　与えられたデータを基にこの範囲の需要を予測

### sell_prices.csv  

### info
店舗数 10(CA:4, Tx:3, WI:3)

### Discussion & knowleadge
似た過去のコンペ
[M5 forecast 2 python](https://www.kaggle.com/kneroma/m5-forecast-v2-python)  
検証はダミーで行う  
テストのlag特徴量を1日ずつ作成してpredict  

[Few thoughts about M5 competition](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/138881)
grid作成  
lagによるnanを削除するか否か　　
loss関数がfrexibleなのでNNも検討  
アンサンブルとスタッキング  

[Back to (predict) the future - Interactive M5 EDA](https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda)
最大投票のEDA

### MEMO
4/8  
データ数や特徴量を追加することで結果が悪くなることが多い 　
ノイズとなっている特徴量がある or leakの可能性  
いずれにしても現在のコードに交差検証の方法を確立しないと方針が立てにくい  

4/9
rmean_lag7_28が圧倒的にgain重要度高い理由について考える  
商品のreleace dayについて 
ある商品だけ周期性がある可能性は十分にあるのでEDAをして見つけたらそれのみを予測するモデルを作る
