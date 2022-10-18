import argparse
import os,sys
from re import A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications import inception_v3 as inc_net, resnet50
from tensorflow.python.ops.gen_math_ops import Max
from White_FGSM import White_Noise

path = os.path.dirname(os.path.dirname('Model/attack_Model/'))
sys.path.append(path)
from Load_Predict import load_and_predict
from NES import Gradient_estimation_fgsm, Gradient_estimation_fgsm_onlyLabel, Gradient_estimation_pgd, lower_upper

class attack_model:
    def __init__(self, net, model, image_size):
        self.net = net
        self.model = model
        self.image_size = image_size

    def fgsm_model(self, args, x_init, ismask, isOnlyLabel):
        """FGSM attack algorithm

        Args:
            args: Various Parameters
            x_init: original image but conform to network input format
            ismask: LIME/CAM or None

        Returns:
            data_result: csv文件格式, head = "init", "adv", "noise"
            x_adv_list: Matrix after adding disturbance
        """
        LP = load_and_predict(self.net, self.model, self.image_size)
        result_list = []
        x_adv_list = []

        for i in range(x_init.shape[0]):
            x_adv = x_init[i]
            init_label = LP.MaxFeatherPredict(x_init[i])
            if ismask == "LIME":
                Mask = LP.mask_extract_lime(args, x_init[i])
            elif ismask == "CAM":
                Mask = LP.mask_extrack_cam(args, x_init[i])
            else:
                Mask = np.zeros([self.image_size, self.image_size, 3])
            if isOnlyLabel == True:
                g = Gradient_estimation_fgsm_onlyLabel(args, x_init[i], init_label, Mask, ismask, self.net, self.model, self.image_size)
            else:
                g = Gradient_estimation_fgsm(args, x_init[i], init_label, Mask, ismask, self.net, self.model, self.image_size)
            for adv_speed in np.arange(0, 0.45, 0.05):
                x_adv = np.clip(x_adv - adv_speed*np.sign(g), 0, 1)
                adv_label_i = LP.MaxFeatherPredict(x_adv)
                if adv_label_i[0][1] != init_label[0][1]:
                    break
            print(i)
            adv_label = LP.MaxFeatherPredict(x_adv)
            noises = np.sqrt(np.sum(np.square(x_adv - x_init[i])))
            rows_dict = {}
            rows_dict.update({'init': init_label[0][1], 'adv': adv_label[0][1], 'noise': noises})
            result_list.append(rows_dict)
            x_adv_list.append(x_adv)
        
        data_result = pd.DataFrame(result_list)
        x_adv_list = np.array(x_adv_list)
        return data_result, x_adv_list

    def pgd_model(self, args, x_init, ismask):
        """PGD attack algorithm

        Args:
            args: Various Parameters
            x_init: original image but conform to network input format
            ismask: LIME/CAM or None

        Returns:
            data_result: csv文件格式, head = "init", "adv", "noise"
            x_adv_list: Matrix after adding disturbance
        """
        LP = load_and_predict(self.net, self.model, self.image_size)
        result_list = []
        x_adv_list = []

        for i in range(x_init.shape[0]):
            x_adv = x_init[i]
            init_label = LP.MaxFeatherPredict(x_init[i])
            if ismask == "LIME":
                Mask = LP.mask_extract_lime(args, x_init[i])
            elif ismask == "CAM":
                Mask = LP.mask_extrack_cam(args, x_init[i])
            else:
                Mask = np.zeros([self.image_size, self.image_size])
            lower, upper = lower_upper(args, x_init[i], Mask, ismask)
            for j in range(args.N):
                x_adv = Gradient_estimation_pgd(args, x_init[i], x_adv, init_label, Mask, lower, upper,
                                                    ismask, self.net, self.model, self.image_size)
                adv_label_i = LP.MaxFeatherPredict(x_adv)
                if adv_label_i[0][1] != init_label[0][1]:
                    break
            print(i)
            adv_label = LP.MaxFeatherPredict(x_adv)
            noises = np.sqrt(np.sum(np.square(x_adv - x_init[i])))
            rows_dict = {}
            rows_dict.update({'init': init_label[0][1], 'adv': adv_label[0][1], 'noise': noises})
            result_list.append(rows_dict)
            x_adv_list.append(x_adv)
            
        data_result = pd.DataFrame(result_list)
        x_adv_list = np.array(x_adv_list)
        return data_result, x_adv_list
    
    def white_fgsm_model(self, args, x_init):
        LP = load_and_predict(self.net, self.model, self.image_size)
        result_list = []
        x_adv_list = []
        for i in range(x_init.shape[0]):
            x_adv = x_init[i]
            init_label = LP.MaxFeatherPredict(x_init[i])
            Mask = LP.mask_extrack_cam(args, x_init[i])
            g = White_Noise(x_adv, Mask)
            for adv_speed in np.arange(0, 0.5, 0.05):
                x_adv = np.clip(x_adv - adv_speed*np.sign(g), 0, 1)
                adv_label_i = LP.MaxFeatherPredict(x_adv)
                if adv_label_i[0][1] != init_label[0][1]:
                    break
            print(i)
            adv_label = LP.MaxFeatherPredict(x_adv)
            noises = np.sqrt(np.sum(np.square(x_adv - x_init[i])))
            rows_dict = {}
            rows_dict.update({'init': init_label[0][1], 'adv': adv_label[0][1], 'noise': noises})
            result_list.append(rows_dict)
            x_adv_list.append(x_adv)

        data_result = pd.DataFrame(result_list)
        x_adv_list = np.array(x_adv_list)
        return data_result, x_adv_list

