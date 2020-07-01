# M5 Forecasting - Accuracy
#### Estimate the unit sales of Walmart retail goods
[competition link](https://www.kaggle.com/c/m5-forecasting-accuracy)

## Leader Board Ranking
- public LB: 1518th  
- private LB: 512th (blonze 🥉)

## competiton contents
warmarketの過去の販売データを用いて、データにない将来の販売需要を予測を行うコンペ  
(Public) stage1では1914日目から1941日目までの予測を行う。  
(Private) stage2(コンペの最終評価)では1942日目から1969日目までの予測を行う。      

## about data
### 1. calendar.csv
2011/01/29 ~ 2016/06/19 までの1969日間のカレンダー情報  

### 2. sales_train_validation.csv  
1913日間(訓練データ)の売り上げ情報  
train: d_1(2011/01/29)  ~ d_1913(2016/04/24)

### 3. sample_submission.csv  
ある匿名のアイテム30490個が未来の28日間で何個売れるかを予測する  
stage1: d_1914(2016/04/25) ~ d_1941(2016/05/22)  
stage2: d_1942(2016/05/23) ~ d_1969(2016/06/19)

### 4. sell_prices.csv  
商品の日ごとの販売価格  

## solution
### 方針
評価指標が独特で交差検証がうまくいかないコンペだったので  
ハイパラは基本的に触らず、学習方法や予測方法を工夫することを意識した  


### モデル  
lightGBMで作成した3つのモデルをアンサンブル  
①お店ごとに訓練を行ったモデル  
②ジャンルごとに訓練したモデル  
③お店のジャンルごとに訓練したモデル  

### validation
1914日目-1941日目(public LBに対応する予測範囲)を対象にhold-out validationを行う  
validation結果が最も良かったものを最終submissionとして選択
  
### feature enginnering
- 1, 7, 14, 30, 60, 180日でlag, rolling特徴量
(1,7,14の特徴量は予測時には通常使えないが以下のrecurisive featuresを予測時に作成することでこれらの特徴量もモデルに利用できるようにした)
- recurisive features-予測時にlag特徴量を作成  
(ただし、public LBの期間に過学習している可能性が高いので有効ではないかも?)  
- 商品の簡易的なリリース日  
(初めの段階では販売されていない商品もあるため)  
- 販売価格の集約統計量   

### predict
1日ごとに予測を行い,前日の予測結果からlag,rolling特徴量を作成し次の日を予測(recurisive)

### その他試したこと
以下は試したがCVが落ちてしまったので利用しなかったこと  
- 売れるか売れないかの2値分類  
- ベースラインのモデルで予測を行うと、商品ごとの需要は全然当てられていなかったが,お店ごと,商品カテゴリごと、エリアごとの日ごとの販売合計数はうまく予測できていたので
ベースラインで予測した販売数を使って日ごとの販売総数を特徴量として追加し再学習  
- 日ごとではなく週ごとの予測  

