English | [简体中文](./README_CN.md)
# ImageGuard

forked from [xqtbox/generalImageClassification](https://github.com/xqtbox/generalImageClassification)

This project is a generalized image classification project, and takes pornographic, political, terrorist and general pictures 4 classifications as examples. The essence of image violation QC is image classification, so the key points are two:

1. data preparation for image classification;
2. image classification model selection, training;

## 1 Data Preparation

In order to achieve the categorization of a specific category, prepare the corresponding image data, the
- 1 Open source dataset
- 2 Crawling the data yourself
- 3 Utilize a specific website (crawler) that will download the data for you.

### 1.1 Open Source Datasets

It would be the happiest thing to start an image-related project in a field where there are public, open-source datasets. So when you have a project requirement, the first thing you can do is to go to github and other websites to search for datasets that can be used directly.

For our “image quality control” project, we can't find ready-made datasets for political and terrorist-related images. However, there are many public datasets for pornographic images, and the quality of the images is very good. Here are two examples:

1. nsfw_data_scrapper public dataset (image address below, and some blogs describing how to use it):
    - nsfw_data_scraper data https://github.com/alex000kim/nsfw_data_scraper 
    - NSFW Model (training resnet using nsfw_data_scrapper data) https://github.com/rockyzhengwu/nsfw
    - nsfw_data_scraper blog https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/86653609
2. nsfw_data_source_urls public dataset:
    - nsfw_data_source_urls: https://github.com/EBazarov/nsfw_data_source_urls (some of ImageGuard's datasets is used in this project)

## 2 Structure

### 2.1 Structure
```
- data: folder with download links to the training set and the validation set
    - train: folder with download links for training images
    - validation: folder with the validation image data
- dataset: folder for the training set
- model: folder for the trained model
- trainModel.py: model training code
- loadDataset.py: download dataset code
- predictImage.py: call the model predictImage code
```

### 2.2 Main packages used

- Pytorch