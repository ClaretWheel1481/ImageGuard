[English](./README.md) | 简体中文
# ImageGuard

forked from [xqtbox/generalImageClassification](https://github.com/xqtbox/generalImageClassification)

本项目是通用的图像分类项目，并以涉黄、涉政、涉恐和普通图片4分类为例。图像违规质检本质是图像分类，所以关键点在于两个：

1. 图像分类的数据准备；
2. 图像分类的模型选择、训练；

## 1 数据准备

为了达到特定类别的分类，准备相应的图片数据，
- 1 开源的数据集
- 2 自己爬取数据
- 3 利用特定的网站（爬虫），帮你下载数据

### 1.1 开源数据集

如果开始一个图像相关的项目，而这个领域又有公开、开源的数据集，那是最幸福的一件事了。所以有了项目需求之后，第一件事情，可以去github等网站搜寻一下有没有可以直接使用的数据集。

而对我们的“图片质检”项目，涉政图片、涉恐图片网上找不到现成的数据集。但是涉黄图片却又很多公开数据集，并且图片质量灰常的“优秀”。下面给出两个实例：

1. nsfw_data_scrapper公开数据集（下面是图片地址，和一些介绍如何使用的博客）：
    - nsfw_data_scraper 数据 https://github.com/alex000kim/nsfw_data_scraper 
    - NSFW Model（使用nsfw_data_scrapper数据训练resnet） https://github.com/rockyzhengwu/nsfw
    - nsfw_data_scraper 博客 https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/86653609
2. nsfw_data_source_urls公开数据集：
    - nsfw_data_source_urls： https://github.com/EBazarov/nsfw_data_source_urls （本项目使用了部分该项目数据）

## 2 项目结构及主要使用的包

### 2.1 项目结构
```
- data：存放训练集下载链接以及验证集文件夹
    - train：存放训练图片下载链接的文件夹
    - validation：存放验证集图片数据的文件夹
- dataset: 存放训练集文件夹
- model: 存放训练后的模型
- trainModel.py: 模型训练代码
- loadDataset.py: 下载数据集代码
- predictImage.py: 调用模型预测图像代码
```

### 2.2 主要使用的包

- Pytorch