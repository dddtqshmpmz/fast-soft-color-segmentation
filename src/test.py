from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from mydataset import MyDataset
import cv2
import time
from guided_filter_pytorch.guided_filter import GuidedFilter
import sys
from train import reconst_loss,mono_color_reconst_loss,squared_mahalanobis_distance_loss,psnr,ssim,sparse_loss
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

def sparse_loss(alpha_layers,device):
    # alpha_layers: bn, ln, 1, h, w
    #print('alpha_layers.mean().item(): ', alpha_layers.mean().item())
    alpha_layers = alpha_layers.sum(dim=1, keepdim=True) / (alpha_layers * alpha_layers).sum(dim=1, keepdim=True)
    loss = F.l1_loss(alpha_layers, torch.ones_like(alpha_layers).to(device))
    return loss

# 必要な関数を定義する
# 用人工挑选的颜色代替之前选择的主要颜色
def replace_color(primary_color_layers, manual_colors):
    temp_primary_color_layers = primary_color_layers.clone()
    for layer in range(len(manual_colors)):
        for color in range(3):
                temp_primary_color_layers[:,layer,color,:,:].fill_(manual_colors[layer][color])
    return temp_primary_color_layers


def cut_edge(target_img):
    #print(target_img.size())
    target_img = F.interpolate(target_img, scale_factor=resize_scale_factor, mode='area')
    #print(target_img.size())
    h = target_img.size(2)
    w = target_img.size(3)
    h = h - (h % 8)
    w = w - (w % 8)
    target_img = target_img[:,:,:h,:w]
    #print(target_img.size())
    return target_img

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w 以层的状态接收并以该形式返回 Bn，ln，1，h，w
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def read_backimage():
    img = cv2.imread('../dataset/backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1)) # c,h,w
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))
    return img.view(1,3,256,256).to(device)

def proc_guidedfilter(alpha_layers, guide_img):
    # guide_imgは， 1chのモノクロに変換 guide_img转换为1通道单色
    # target_imgを使う． bn, 3, h, w
    guide_img = (guide_img[:, 0, :, :]*0.299 + guide_img[:, 1, :, :]*0.587 + guide_img[:, 2, :, :]*0.114).unsqueeze(1)
        
    # lnのそれぞれに対してguideg filterを実行 对ln的每个运行引导过滤器
    for i in range(alpha_layers.size(1)):
        # layerは，bn, 1, h, w
        layer = alpha_layers[:, i, :, :, :]
        
        processed_layer = GuidedFilter(3, 1*1e-6)(guide_img, layer) #可以去了解一下什么是GuidedFilter smooth the image
        # レイヤーごとの結果をまとめてlayersの形に戻す (bn, ln, 1, h, w)
        # 将每个图层的结果放回图层（bn，ln，1，h，w）
        if i == 0: 
            processed_alpha_layers = processed_layer.unsqueeze(1)
        else:
            processed_alpha_layers = torch.cat((processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1)
    
    return processed_alpha_layers

## Define functions for mask operation.
# マスクを受け取る関数
# target_layer_numberが冗長なレイヤーの番号（２つ）のリスト．これらのレイヤーに操作を加える
# 接收蒙版的功能
# 层编号（2）的列表，其中target_layer_number是冗余的。 向这些层添加操作

def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0) #白黒で読み込み ＃黑白阅读
    mask[mask<128] = 0.
    mask[mask >= 128] = 1.
    # tensorに変換する #转换为张量
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
    
    return mask

def mask_operate(alpha_layers, target_layer_number, mask_path):
    layer_A = alpha_layers[:, target_layer_number[0], :, :, :]
    layer_B = alpha_layers[:, target_layer_number[1], :, :, :]
    processed_alpha_layers
    layer_AB = layer_A + layer_B
    mask = load_mask(mask_path)
    
    mask = cut_edge(mask)
    
    layer_A = layer_AB * mask
    layer_B = layer_AB * (1. - mask)
    
    return_alpha_layers = alpha_layers.clone()
    return_alpha_layers[:, target_layer_number[0], :, :, :] = layer_A
    return_alpha_layers[:, target_layer_number[1], :, :, :] = layer_B
    
    return return_alpha_layers

#### User inputs
run_name = 'train_0927'
num_primary_color = 7
csv_path = 'train.csv' # なんでも良い．後方でパスを置き換えるから
csv_path_ihc = 'train_IHC_256_2w.csv'
csv_path_test = 'train_IHC.csv'

# 设置loss权重
rec_loss_lambda = 1.0
m_loss_lambda = 1.0
sparse_loss_lambda = 0.0
distance_loss_lambda = 0.5

# 设置哪台机器上跑test
device_id = 1
device = 'cuda:'+ str(device_id)

# sys.stdout = sys.__stdout__
# 打印所有数据到日志
log = open("test_%s.log" % (run_name), "a")
sys.stdout = log

resize_scale_factor = 1  

path_mask_generator = 'results/' + run_name + '/mask_generator.pth'
path_residue_predictor = 'results/' + run_name + '/residue_predictor.pth'


test_dataset = MyDataset(csv_path, csv_path_ihc, csv_path_test,num_primary_color, mode='test')
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    )

# define model
mask_generator = MaskGenerator(num_primary_color).to(device)
residue_predictor = ResiduePredictor(num_primary_color).to(device)
# 注意使用多gpu并行训练 需要添加如下代码
mask_generator = nn.DataParallel(mask_generator,device_ids=[device_id])
mask_generator = mask_generator.cuda(device)
residue_predictor = nn.DataParallel(residue_predictor,device_ids=[device_id])
residue_predictor = residue_predictor.cuda(device)

# load params
mask_generator.load_state_dict(torch.load(path_mask_generator))
residue_predictor.load_state_dict(torch.load(path_residue_predictor))

# eval mode
mask_generator.eval()
residue_predictor.eval()

backimage = read_backimage()

target_layer_number = [0, 1] # マスクで操作するレイヤーの番号 ＃层号与遮罩一起使用
mask_path = 'path/to/mask.image'


print('Start!')

mean_estimation_time = 0
img_index = 0
with torch.no_grad():
    test_loss = 0   
    r_loss_mean = 0
    m_loss_mean = 0
    s_loss_mean = 0
    d_loss_mean = 0
    psnr_mean = 0
    ssim_mean = 0

    for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
        print('img #', batch_idx)

        target_img = cut_edge(target_img)
        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)

        start_time = time.time()
        primary_color_pack = primary_color_layers.view(primary_color_layers.size(0), -1 , primary_color_layers.size(3), primary_color_layers.size(4))
        primary_color_pack = cut_edge(primary_color_pack)
        primary_color_layers = primary_color_pack.view(primary_color_pack.size(0),-1,3,primary_color_pack.size(2), primary_color_pack.size(3))
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
        ## Alpha Layer Proccessing
        processed_alpha_layers = alpha_normalize(pred_alpha_layers) 
        #processed_alpha_layers = mask_operate(processed_alpha_layers, target_layer_number, mask_path) # Option
        processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img) # Option
        processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Option
        ##
        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
        residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

        end_time = time.time()
        estimation_time = end_time - start_time
        print('estimation_time: ',estimation_time)
        mean_estimation_time += estimation_time


        # 计算loss 打印出来
        r_loss = reconst_loss(reconst_img, target_img, 'l1') * rec_loss_lambda
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * m_loss_lambda
        s_loss = sparse_loss(processed_alpha_layers,device) # 注意这个没有乘以 权重
        #print('total_loss: ', total_loss)
        d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * distance_loss_lambda
        psnr_res = psnr(reconst_img, target_img)
        ssim_res = ssim(reconst_img, target_img)

        total_loss = r_loss + m_loss + s_loss * sparse_loss_lambda + d_loss
        test_loss += total_loss.item()
        r_loss_mean += r_loss.item()
        m_loss_mean += m_loss.item()
        s_loss_mean += s_loss.item()
        d_loss_mean += d_loss.item()

        psnr_mean += psnr_res.item()
        ssim_mean += ssim_res.item()

        print('r_loss:', r_loss)
        print('psnr:', psnr_res)
        print('ssim:', ssim_res)
        print('sparsity:',s_loss)

        if (batch_idx %10 == 0 and True): # 
            img_index = 1
            try:
                os.makedirs('results/%s/test' % (run_name))
            except OSError:
                pass
            save_layer_number = 0
            save_image(primary_color_layers[save_layer_number,:,:,:,:],
                   'results/%s/test/test' % (run_name) + '_img-%02d_primary_color_layers.png' % batch_idx)
            save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/test/test' % (run_name)  + '_img-%02d_reconst_img.png' % batch_idx)
            save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/test/test' % (run_name)  + '_img-%02d_target_img.png' % batch_idx)

            '''
            # RGBAの４chのpngとして保存する 另存为RGBA 4ch png
            RGBA_layers = torch.cat((pred_unmixed_rgb_layers, processed_alpha_layers), dim=2) # out: bn, ln, 4, h, w
            # test ではバッチサイズが１なので，bn部分をなくす 在测试中，批量大小为1，因此消除了bn部分。
            RGBA_layers = RGBA_layers[0] # ln, 4. h, w
            # ln ごとに結果を保存する
            for i in range(len(RGBA_layers)):
                save_image(RGBA_layers[i, :, :, :], 'results/%s/test/img-%02d_layer-%02d.png' % (run_name, batch_idx, i) )


            # 処理まえのアルファを保存 处理前保存Alpha
            for i in range(len(pred_alpha_layers[0])):
                save_image(pred_alpha_layers[0,i, :, :, :], 'results/%s/test/pred-alpha-00_layer-%02d.png' % (run_name, i) )

            # 処理後のアルファの保存 processed_alpha_layers 保存已处理的alpha已处理的alpha_layers
            for i in range(len(processed_alpha_layers[0])):
                save_image(processed_alpha_layers[0,i, :, :, :], 'results/%s/test/proc-alpha-00_layer-%02d.png' % (run_name, i) )

            # 処理後のRGBの保存 处理后保存RGB
            for i in range(len(pred_unmixed_rgb_layers[0])):
                save_image(pred_unmixed_rgb_layers[0,i, :, :, :], 'results/%s/test/rgb-00_layer-%02d.png' % (run_name, i) )
            '''
            print('Saved to results/%s/test/...' % (run_name))
           
        

    test_loss = test_loss / len(test_loader.dataset)
    r_loss_mean = r_loss_mean / len(test_loader.dataset)
    m_loss_mean = m_loss_mean / len(test_loader.dataset)
    s_loss_mean = s_loss_mean / len(test_loader.dataset)
    d_loss_mean = d_loss_mean / len(test_loader.dataset)
    
    psnr_mean = psnr_mean / len(test_loader.dataset)
    ssim_mean = ssim_mean / len(test_loader.dataset)

    print('====> Average test loss: {:.6f}'.format(test_loss ))
    print('====> Average test reconst_loss *lambda: {:.6f}'.format( r_loss_mean ))
    print('====> Average test mono_loss *lambda: {:.6f}'.format( m_loss_mean ))
    print('====> Average test squared_mahalanobis_distance_loss *lambda: {:.6f}'.format( d_loss_mean ))

    print('====> Average test psnr: {:.6f}'.format( psnr_mean ))
    print('====> Average test ssim: {:.6f}'.format( ssim_mean ))
    print('====> Average test sparse_loss: {:.6f}'.format( s_loss_mean ))

    print('mean_estimation_time: ', mean_estimation_time / len(test_loader.dataset))
    
