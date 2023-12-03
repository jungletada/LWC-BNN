# Bayesian Neural Network for Liquid Water Content Retrieval Using the Differences in Radar Attenuation

## Structure of Data
- main path for all data: `data/` 
- Five dates for experiments `data/{date}/`
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
- 运行`run_mcmc.py`, 将结果保存至`results/pred.npy`中   
    - `python run_bnn.py`
- 运行`visualize.py`, 可视化预测结果
    - `python visualize.py`
    
 ## Data Statistic 
