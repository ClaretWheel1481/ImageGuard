import os
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# 多线程下载数据集图片
# 创建文件夹
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_image(url, save_dir):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img_name = os.path.basename(urlparse(url).path)
        img_path = os.path.join(save_dir, img_name)
        with open(img_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {img_name}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_dataset(urls_file, save_dir):
    create_directory(save_dir)
    with open(urls_file, 'r') as f:
        urls = f.read().splitlines()
    with ThreadPoolExecutor(max_workers=100) as executor:
        for url in urls:
            executor.submit(download_image, url, save_dir)

def download_all_datasets():
    datasets = ['neutral', 'porn']
    for dataset in datasets:
        urls_file = f'data/train/{dataset}/urls_{dataset}.txt'
        save_dir = f'dataset/{dataset}'
        download_dataset(urls_file, save_dir)

download_all_datasets()