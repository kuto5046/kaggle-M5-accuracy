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
train: d_1 ~ d_1913

### sample_submission.csv
ある匿名のアイテム827個が未来の28日間で何個売れるかを予測する  
valid: d_1914 ~ d_1941
eval: d_1942 ~ d_1969

### sell_prices.csv  

### info
店舗数 10(CA:4, Tx:3, WI:3)

### Discussion & knowleadge
(2020/3/30)  
- 
