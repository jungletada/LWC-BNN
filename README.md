# Estimating Liquid Water Content Using Dual-Frequency Radar and Bayesian Neural Networks
Zheng Wen∗, Dingjie Peng∗, Xun Su∗, Yousuke Ohya∗, Kazuhiko Tamesue∗, Hiroyuki Kasai∗, Wataru Kameyama∗, Takuro Sato∗
## Structure of Data
- main path for all data: `data-slim/` 
- for experiments `data-slim/{date}/`
    - `{date}_ka_band.csv`: Ka-band雷达反射因子
    - `{date}_w_band.csv`: W-band雷达反射因子
    - `{date}_pressure.csv`: 湿度
    - `{date}_relativeHumidity.csv`: 相对湿度
    - `{date}_temperature.csv`: 温度
    - `{date}_lwc.csv`: 云水量(预测目标)
  
- 去除掉了高度600米以下的所有数据（高度阈值为600，因为600米以下的没有Ka-band数据，也没有云水量）
- 将Ka-band和w-band的数据，按照云水量（LWC）的高度进行了线性插值，目前所有对应日期的数据的shape一致（即高度值，时间戳数量一致）

## How to use  
本项目使用MLP构建贝叶斯神经网络。MLP的模型定义位于`models/`文件夹中
- 运行`main_bnn.py`, 将结果保存至`results`中   
    - `python run_bnn.py`
- 直接运行脚本 `bash run.sh`

## 结果
所有结果保存在`results/`.

## 论文专用画图
20130515的全天LWC预测  
- MCMC-MLP: results/mcmc-mlp/mcmc_pred.csv
- Random Forest: results/random-forest/random-forest_pred.csv
- Linear-regression: results/linear-regression/linear-regression_pred.csv
- Decision-tree: results/decision-tree/decision-tree_pred.csv
- Xgboost: results/xgboost/xgboost_pred.csv
## 评估指标结果保存到excel文件  
results/model_evaluation.xlsx

## 日本語の説明
- 高度600メートル以下のデータをすべて除外しました（600メートル以下にはKaバンドのデータも雲水量もありません）。
- KaバンドとWバンドのデータを、雲水量（LWC）の高度に基づいて線形補間しました。現在、すべての対応する日付のデータの形状が一致しています（つまり、高度値とタイムスタンプの数が一致しています）。

### 機械学習モデル
`main_ml.py`というファイルを使って、4つの機械学習モデルが含まれています。

### データセットを構築する方法
`data_loader.py`の手法にならって、CSVファイルを読み込み、データの入力形状を（#timestamps、#dimension）に変換することです。 
入力データの構造は（#timestamps、#dimension）であり、出力データの構造は（#timestamps、1）です。その中で1は、対応する時刻のLWC（液体水含有量）の値を表しています。

- #timestamps：タイムスタンプと高度の数
- #dimension：入力の次元数を表しています。例えば、ka、w、p、t、h、heightが入力される場合、次元数は6になります。