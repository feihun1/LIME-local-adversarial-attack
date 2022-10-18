# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import PIL
import matplotlib.pyplot as plt
import scipy.io as scio


# input image
LABELS_file = '/Users/duanfeihun/.keras/models/imagenet_class_index.json'

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

# load test image

def CAM_region(min_region, x_init):

    '''
    # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
    if model_id == 1:
        net = models.squeezenet1_1(pretrained=True)
        finalconv_name = 'features' # this is the last conv layer of the network
    elif model_id == 2:
        net = models.resnet101(pretrained=True)
        finalconv_name = 'layer4'
    elif model_id == 3:
        net = models.densenet161(pretrained=True)
        finalconv_name = 'features'
    '''
    net = models.resnet101(pretrained=True)
    #net = models.inception_v3(pretrained=True)
    finalconv_name = 'layer4'
    net.eval()

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    x_init = x_init * 255

    img_pil = Image.fromarray(np.uint8(x_init))
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # load the imagenet category list
    with open(LABELS_file) as f:
        classes = json.load(f)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    
    height, width, _ = x_init.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), 4)
    #result = heatmap * 0.3 + x_init[i] * 0.5
    #cv2.imwrite('test1.jpg', result)
    #heatmap = heatmap / 255

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i][j][0] != 0:
                heatmap[i][j]=[1,1,1]
            elif heatmap[i][j][1] > min_region:
                heatmap[i][j]=[1,1,1]
            else:
                heatmap[i][j]=[0,0,0]

    return heatmap[:,:,-1]