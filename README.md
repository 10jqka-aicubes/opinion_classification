# 【2-4春季赛】高度口语化强背景知识的文本观点分类

​        用户舆情观点检测（User Public Opinion Detection）是目前自然语言处理领域的重点应用方向。随着社交媒体普及，用户发布的实时观点信息存在大量可用信息。在投顾场景中可通过用户观点信息分类辅助投资者做出投资决策。目前用户舆情检测在应用中往往会面临以下挑战：1.用户表述偏口语化甚至不能准确描述问题；2.用户的情感表达往往个性化，文本上很难捕捉；3.数据样本标注标准上存在主观差异性。

- 本代码是该赛题的一个基础demo，参考[Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA)，仅供参考学习


- 比赛地址：http://contest.aicubes.cn/	


- 时间：2022-02 ~ 2022-04

  ​

## 如何运行Demo

- clone代码


- 预训练模型下载，存放在参数`model_data_dir`对应的路径下

  ```
  https://github.com/ymcui/Chinese-ELECTRA -> ELECTRA-small, Chinese Tensorflow
  ```

- 准备环境

  - cuda10.0以上
  - python3.7以上
  - 安装python依赖

  ```
  python -m pip install -r requirements.txt
  ```

- 准备数据，从[官网](http://contest.aicubes.cn/#/detail?topicId=47)下载数据

  - 训练集存放在`train/run.sh`的`--raw_data_dir`对应的路径下
  - 测试集重命名为`test.txt`，存放在`predict/run.sh`的`--raw_data_dir`对应的路径下

- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明，主要配置文件`opinion_classification/setting.conf`

- 运行

  - 训练

  ```
  bash opinion_classification/train/run.sh
  ```

  - 预测

  ```
  bash opinion_classification/predict/run.sh
  ```

  - 计算结果指标

  ```
  bash opinion_classification/metrics/run.sh
  ```

## 反作弊声明

1）参与者不允许在比赛中抄袭他人作品、使用多个小号，经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码。



## 赛事交流

![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)