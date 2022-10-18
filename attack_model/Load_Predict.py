#Load the image data and predict the max label
import numpy as np
import os,sys
from os import path
from numpy.lib.type_check import imag
import glob
import scipy.io as scio
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions

path = os.path.dirname(os.path.dirname('Model/attack_Model/'))
sys.path.append(path)
from CAM import CAM_region

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

explainer = lime_image.LimeImageExplainer() #Lime处理图像类的实例化

class load_and_predict:

    def __init__(self, net, model, image_size):
        self.net = net
        self.model = model
        self.image_size = image_size
        
    def transform_img_fn(self, path_list):
        #图像格式处理
        """

        Args:
            path_list: Input path of image

        Returns:
            out: Processed image list 
        """
        out = []
        for img_path in path_list:
            img = image.load_img(img_path, target_size=(self.image_size, self.image_size))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            out.append(x)
        return np.vstack(out)

    def MaxFeatherPredict(self, images):
        #图片最大可能类别预测
        """Most likely prediction results of inception V3 network

        Args:
            images: pictures that meet the input requirements

        Returns:
            Most likely categories
        """
        img = images.copy()
        img = img * 255.0
        img = self.net.preprocess_input(img)
        if img.shape != (1, self.image_size, self.image_size, 3):
            img = np.expand_dims(img, axis=0)
        preds = self.model.predict(img)
        return decode_predictions(preds)[0]

    def load_image(self, data_path):
        #图片和对应的重要特征的加载
        """ Load the images and import features in the dataset 

        Args:
            data_path: the path of dataset

        Returns:
            x_init: Original images list
        """
        data_dir = path.join('/Users/duanfeihun/Desktop/AI与安全/对抗攻击/LIME/dataset/' + data_path)
        categories = os.listdir(data_dir)
        if '.DS_Store' in categories:
            categories.remove('.DS_Store')

        image_path = []
                
        for i in range(100):
            category = categories[i]
            cat_dir = path.join(data_dir, category)
            images = os.listdir(cat_dir)

            for j in range(2):
                image_path.append(cat_dir + '/' + images[j])

        x_init = self.transform_img_fn(image_path)
        return x_init

    def mask_extract_lime(self, args, images):
        #利用LIME方法提取重要特征
        """Extracting important features using LIME method

        Args:
            args: Various Parameters
            images: original image

        Returns:
            maask: Important characteristic matrix
        """
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images.astype('double'), self.model.predict, top_labels = args.top_k, hide_color=0, num_samples=600)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features = args.lime_region, hide_rest=False)
        return mask

    def mask_extrack_cam(self, args, images):
        heatmap = CAM_region(args.min_pixel, images)
        return heatmap
