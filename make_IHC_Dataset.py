import numpy as np
import cv2
import os
import pandas as pd
import csv
import glob
from sklearn.cluster import KMeans
import random

# ihc 大图片中采样小图片 制作新数据集 or 补充数据集
def getIHCdataset():
    img_size = 256
    sample_num = 40 # 每张大图片中采样小图片的数量
    img_ids = glob.glob(os.path.join('dataset','train_IHC', '*' +'.png'))
    for img_id in img_ids:
        img = cv2.imread(img_id)
        h,w,c = img.shape
        for i in range(sample_num):
            sample_img_id = 'dataset/train_IHC_256_2w/' + img_id.split('/')[-1].split('.')[0] + '_' + str(i) + '.png'
            left_top_x = random.randint(0,w-img_size-1)
            left_top_y = random.randint(0,h-img_size-1)
            cv2.imwrite(sample_img_id,img[left_top_y : left_top_y+img_size, left_top_x : left_top_x+img_size])

# ihc 采样数据集 生成图片名+调色盘csv
def getNameAndPaleteCSV():
    train_img_ids = glob.glob(os.path.join('dataset','train_IHC_256_2w', '*' +'.png'))
    num_clusters = 7
    idx = 0
    for train_img_id in train_img_ids:
        idx += 1
        print(idx)
        img = cv2.imread(train_img_id)[:, :, [2, 1, 0]]
        size = img.shape[:2]
        vec_img = img.reshape(-1, 3)
        model = KMeans(n_clusters=num_clusters, n_jobs=-1)
        pred = model.fit_predict(vec_img)
        center = model.cluster_centers_.reshape(-1)
        print(np.floor(center))
        with open("train_IHC_256_2w.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['../' + train_img_id])
        with open("pallette_7_train_IHC_256_2w.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(np.floor(center))


    print('hello')

if __name__ == '__main__':
    # getIHCdataset()
    # getNameAndPaleteCSV()

    print('hello')