import numpy as np
import tensorflow as tf
import os,sys
path = os.path.dirname(os.path.dirname('Model/attack_Model/'))
sys.path.append(path)
from Load_Predict import load_and_predict


def lower_upper(args, x_init, Mask, ismask):
    #使用PGD攻击，需要对重要特征控制每次迭代添加扰动的大小
    """If you want to use PGD attack, you must control the step size of each iteration

    Args:
        x_init: Original image
        Mask: Important feature matrix

    Returns:
        x_init_lower: Anti-sample perturbation lower bound
        x_init_upper: Anti-sample perturbation upper bound
    """
    if ismask == "LIME" or "CAM":
        mask = Mask.copy()
        mask = mask[:, :, np.newaxis]
        mask = np.tile(mask, 3)
        x_init_lower = np.clip(x_init - (mask * args.epsilon), 0, 1)
        x_init_upper = np.clip(x_init + (mask * args.epsilon), 0, 1)
    else:
        x_init_lower = np.clip(x_init - args.epsilon, 0, 1)
        x_init_upper = np.clip(x_init + args.epsilon, 0, 1)
    return x_init_lower, x_init_upper

def Noise_Add(args, image, Mask, ismask):
    #在对重要特征进行扰动时，可以将添加噪声的这一部分代码封装成一个函数
    """ When disturbing important features, you can encapsulate this part of the code that adds noise into a function

    Args:
        image: Original image in round k
        Mask: Important feature matrix
        n: size of each NES population 
        sigma: Add the intensity of Gaussian noise to the image

    Returns:
        init: Image after adding noise
        noise: Gaussian noise
        
    """
    x_adv_array = np.array([image for i in range(args.n)])
    noise_pos = np.array(tf.random.normal((args.n//2,) + image.shape))
    if ismask == "LIME" or "CAM":
        mask = Mask.copy()
        mask = mask[:, :, np.newaxis]
        mask = np.tile(mask, 3)
        for i in range(noise_pos.shape[0]):
            noise_pos[i] *= mask
    noise = np.concatenate((noise_pos, -noise_pos), axis=0)
    init = x_adv_array + args.sigma * noise
    return init, noise

def Pro_Es(args, image, Mask, ismask, net, model, image_size):
    LP = load_and_predict(net, model, image_size)
    first_label = LP.MaxFeatherPredict(image)[0]
    init, noise = Noise_Add(args, image, Mask, ismask)

    num_index = 0
    for i in range(init.shape[0]):
        label = LP.MaxFeatherPredict(init[i])
        for j in range(args.top_k):
            if label[j][1] == first_label[1]:
                num_index += args.top_k - j
    p = num_index / args.m
    return p

def Gradient_estimation_pgd(args, x_init, x_adv, y, Mask, lower, upper, ismask, net, model, image_size):
    #使用PGD方法添加噪声
    """Disturb the image

    Args:
        args: Various Parameters
        x_init: Original image
        x_adv: Original image in round k
        y: Label of the original sample
        Mask: Binary matrix, 1 for important features, 0 for unimportant features
        lower: Anti-sample perturbation lower bound
        upper: Anti-sample perturbation upper bound
        ismask: LIME or NotLIME
        net: Attacked network
        model: Attacked model
        image_size: image size conform to network input format

    Returns:
        x_adv: The image with noise added in the kth round is also the original image of the k+1 round
    """
    g = np.zeros([image_size, image_size, 3]).astype(np.float)
    if ismask == "GLOBAL":
        Mask = np.zeros((image_size, image_size, 3))
    init, noise = Noise_Add(args, x_adv, Mask, ismask)
    LP = load_and_predict(net, model, image_size)

    for i in range(init.shape[0]):
        k = LP.MaxFeatherPredict(init[i])
        res = list(filter(lambda x: y[0][1] in x, k))
        g += res[0][2] * noise[i] if i < args.n / 2 else -res[0][2] * noise[i]

    g = np.sign(g)
    x_adv = np.clip(x_adv - args.eta*g, x_init - lower, x_init + upper)
    return x_adv

def Gradient_estimation_fgsm(args, x_init, y, Mask, ismask, net, model, image_size):
    g = np.zeros([image_size, image_size, 3]).astype(np.float)
    init, noise = Noise_Add(args, x_init, Mask, ismask)
    LP = load_and_predict(net, model, image_size)

    for i in range(init.shape[0]):
        k = LP.MaxFeatherPredict(init[i])
        res = list(filter(lambda x: y[0][1] in x, k))
        g += res[0][2] * noise[i] if i < args.n / 2 else -res[0][2] * noise[i]

    g = np.sign(g)
    return g

def Gradient_estimation_fgsm_onlyLabel(args, x_init, y, Mask, ismask, net, model, image_size):
    g = np.zeros([image_size, image_size, 3]).astype(np.float)
    init, noise = Noise_Add(args, x_init, Mask, ismask)
    for i in range(init.shape[0]):
        p = Pro_Es(args, x_init, Mask, ismask, net, model, image_size)
        g += p * noise[i] if i < args.n / 2 else -p * noise[i]
    g = np.sign(g)
    return g