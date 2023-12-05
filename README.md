# Bayesian Neural Network for Liquid Water Content Retrieval Using the Differences in Radar Attenuation

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

## 目前最佳
### MCMC-MLP

- num_samples=250  
- BNN(in_dim=7, out_dim=1, hid_dim=96, n_hid_layers=2, prior_scale=5.)  
Mean Squared Error: 0.00003
Mean Absolute Error: 0.00358
R-squared Score: 0.88981
Explained Variance Score: 0.89109

### Random Forest
Mean Squared Error: 0.00005
Mean Absolute Error: 0.00344
R-squared Score: 0.82427
Explained Variance Score: 0.82718