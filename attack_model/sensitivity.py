#敏感性分析
#数据集：256_ObjectCategories
#网络：VGG16
#块数范围：4-24
#阈值范围：0.1-0.7

import argparse
import os, sys
import scipy.io as scio
import numpy as np

path = os.path.dirname(os.path.dirname('Model/attack_Model/'))
sys.path.append(path)
from attack import attack_model
from Load_Predict import load_and_predict
from tensorflow.keras.applications import inception_v3 as inc_net

from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications import resnet50

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16

from tensorflow.python.keras.applications.xception import Xception
from tensorflow.keras.applications import xception

N = 50
n = 100
m = 50
SIGMA = 0.001
ETA = 0.05
EPSILON = 0.5
top_k = 5

inet_model = inc_net.InceptionV3()
res_model = ResNet50()
vgg_model = VGG16()
xcep_model = Xception()


def Parser_Setting():
    #参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type = int, default = N)
    parser.add_argument('--n', type = int, default = n)
    parser.add_argument('--m', type = float, default = m)
    parser.add_argument('--sigma', type = float, default = SIGMA)
    parser.add_argument('--eta', type = float, default = ETA)
    parser.add_argument('--epsilon', type = float, default = EPSILON)
    parser.add_argument('--top_k', type = int, default = top_k)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = Parser_Setting()
    
    ObjCat_101_Inception = attack_model(inc_net, inet_model, image_size = 299)
    ObjCat_101_ResNet = attack_model(resnet50, res_model, image_size = 224)
    ObjCat_256_VGG16 = attack_model(vgg16, vgg_model, image_size = 224)
    ILSVRC_Xception = attack_model(xception, xcep_model, image_size = 299)

    LP_ObjCat_Xception = load_and_predict(ILSVRC_Xception.net, ILSVRC_Xception.model, ILSVRC_Xception.image_size)
    #x_init = LP_ObjCat_Xception.load_image(data_path = 'ILSVRC2012_1')
    dataFile = '/Users/duanfeihun/Desktop/Model/data/x_init.mat'
    x_init = scio.loadmat(dataFile)['init']
    
    #data_result, x_adv_list = ObjCat_101_Inception.pgd_model(args, x_init, ismask = True, temp = 4)
    #data_result, x_adv_list = ObjCat_101_ResNet.pgd_model(args, x_init, ismask = True, temp = 4)
    data_result, x_adv_list = ObjCat_256_VGG16.pgd_model(args, x_init, ismask = True, temp = 24)
    data_result.to_csv("sensitivity/pgd/" + str(24) + ".csv", index=False)
    Mask_UI = "sensitivity/pgd/" + str(24) + ".mat"
    scio.savemat(Mask_UI, {'adv': x_adv_list})
