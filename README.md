# DE-DAE-MD-Data-Interpolation
This is a repository for the dissertation: Enhancing Data Interpolation with Dimension-Enhanced Denoising Autoencoder with Multi-Decoder (DE-DAE-MD) on the French Building Energy Performance Dataset. 详细的实验思路以及结果分析请见论文。这个README文件主题提供代码以及代码的复现方法。

## 目录
* [Environment](##Environment)
    * Local
    * Colab
* [Dataset](##Dataset)
    * Source
    * Other download links
    * Link for the dataset description files
* [Model](##Model)
    * DAE(baseline model)
    * DE-DAE
    * DE-DAE-MD
    * Ensemble
* [Result](##Result)
    * 来源于网络的图片
    * GitHub仓库中的图片
* [链接](#链接) 
    * 文字超链接
        *  链接外部URL
        *  链接本仓库里的URL
    *  锚点
    * [图片链接](#图片链接)
* [列表](#列表)
    * 无序列表
    * 有序列表
    * 复选框列表
* [块引用](#块引用)
* [代码高亮](#代码高亮)
* [表格](#表格) 
* [表情](#表情)
* [diff语法](#diff语法)



## Environment
### Local
OS: Win 10/11
Frame: Tensorflow
GPU：Nvidia 3080Ti
可以使用如下命令配置环境：
```
pip install -r requirements.txt
```


### Colab
推荐使用Google Colab. 运行时的推荐配置为高RAM + Nvidia A100.


## Dataset
### Source
The research data comes from a dataset in the open database of the French Environment and Energy Management Agency (ADEME): Energy Performance Diagnostics of Existing Housing. The link is shown below :
```
https://data.ademe.fr/datasets/dpe-v2-logements-existants
```
开源数据库网站中提供了多种数据格式，下载时请选择csv格式。下载步骤如下所示：

### Quick download link for the dataset
如果您的数据集下载速度较慢，可以使用下面的链接进行数据集的下载：
```

```
但是需要注意，本链接中的数据最后更新于2023年5月。如果您希望获取包含最新数据的数据集，请按照上面的步骤进行下载。

### Link for the dataset description files
如果您想要详细了解数据集中各个变量的具体信息（包括某些变量的计算方式），请下载如下文档：
```

```




## Model
这个repository提供了论文中所有模型代码，模型主要分为以下四类：
### 1. DAE(Baseline Model)
该模型为传统的DAE，由一个编码器和一个解码器组成。
### 2. Dimension Enhanced DAE
Dimension Enhanced DAE(DE-DAE) 对传统的DAE做了改进，增大了隐空间的维度。DE-DAE可以包含更多的信息，更加有利于数据插补。
### 3. DE-DAE with Multi-decoder
DE-DAE with Multi-decoder(DE-DAE-MD)对DE-DAE进行了进一步的改进。其增加了一些并行的解码器，每个解码器负责不同数据类型的解码。这样做可以增强解码器的数据还原能力。
### 4. Ensemble Model
为了进一步增强模型的数据插补能力，我们采用了Ensemble方法。我们修改了DE-DAE-MD中的激活函数。在Ensemble Model中，我们采用了三个激活函数不同的DE-DAE-MD模型，并且将其结果的平均值作为最终的预测结果。

## Result
由于代码中使用了Dropout层，并且训练数据是随机划分的，这导致每次的训练和测试结果不会完全一致。因此，您如果对代码进行复现，那么出现2%以内的误差是合理的。
实验论文中的结果保存在xxx文件中。您可以点击文件查看其中的结果。

## Document Description
### 目录结构说明
### 









