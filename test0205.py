import numpy as np
import cv2
import os
import pandas as pd
import csv
import glob
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def generate_ihc_csv():
    imgs_dir = '../unet_amax2/unet/inputs/patient_datasetRandom/images_10pics_30k_patches'
    img_ids = glob.glob(os.path.join(imgs_dir, '*' +'.png'))
    for img_id in img_ids:
        with open("ihc_30k.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([img_id])

def generate_palatte_ihc_csv():
    imgs_dir = '../unet_amax2/unet/inputs/patient_datasetRandom/images_10pics_30k_patches'
    train_img_ids = glob.glob( os.path.join( imgs_dir, '*' +'.png'))
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
        with open("ihc_30k.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([train_img_id])
        with open("palette_7_ihc_30k.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(np.floor(center))


if __name__ == '__main__':
    # img_path = 'src/results/sample/1808870-A1-CD4thumb.png/img-00_layer-00.png'
    # img = cv2.imread(img_path,-1)

    # df = pd.read_csv('palette_0607.csv',header=0,index_col=0)
    # df = df.dropna(axis=0,how='all')
    # df.to_csv('palette_7_train.csv')

    # df = pd.read_csv('palette_raw_train.csv',header=0,index_col=0)
    # df.to_csv('palette_7_train.csv',index=0, header=0)

    # df = pd.read_csv('palette_raw_train.csv',header=0,index_col=0)
    # preindex = ''
    # for index, row in df.iterrows():
    #     if len(preindex)>10 and preindex!=index:
    #         pic_name = glob.glob(os.path.join('dataset','train',preindex[0:14]+'*'+'.png'))
    #         pic_name[0] = '../' + pic_name[0]
    #         assert(len(pic_name)==1)
    #         with open("train.csv","a+") as csvfile: 
    #             writer = csv.writer(csvfile)
    #             writer.writerow([pic_name[0]])
    #     elif preindex==index:
    #         continue
    #     preindex = index

    # df = pd.read_csv('palette_raw_train.csv',header=0,index_col=0)
    # preindex = ''
    # for index, row in df.iterrows():
    #     if len(preindex)>10 and preindex!=index:
    #         with open("palette_7_train.csv","a+") as csvfile: 
    #             writer = csv.writer(csvfile)
    #             writer.writerow(row)
    #     elif preindex==index:
    #         continue
    #     preindex = index
    
    # 根据kmeans 生成 365数据集的调色盘标注
    '''
    train_img_ids = glob.glob(os.path.join('dataset','train', '*' +'.jpg'))
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
        with open("train.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['../' + train_img_id])
        with open("palette_7_train.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(np.floor(center))
    '''
    #df = np.array(pd.read_csv('train_IHC.csv',header=None))
    
    # 根据csv 图片 按顺序生成 7色调色盘 csv文件 
    # generate_palatte_ihc_csv()
    
    # --》test 出 一张image 对应 7张rgba图片 编号_0 _1 ... _6 

    # 读取 输入的dataset 
    # 改 unet 语义分割的模型 
    # 跑起来 测试效果

