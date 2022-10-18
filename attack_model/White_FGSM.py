import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np

pretrained_model = tf.keras.applications.ResNet50()
decode_predictions = tf.keras.applications.resnet50.decode_predictions
pretrained_model = tf.keras.applications.InceptionV3()
decode_predictions = tf.keras.applications.inception_v3.decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()


def preprocess(image):
    image = tf.image.resize(image, (299, 299))
    image = image[None, ...]
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]

def create_adversarial_pattern(input_image, input_label, Mask):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
    
    gradient = tape.gradient(loss, input_image)
    mask = Mask.copy()
    mask = mask[:, :, np.newaxis]
    mask = np.tile(mask, 3)
    grad = gradient[0] * mask
    return grad

def White_Noise(x_init, Mask):
    x_init = preprocess(x_init)
    image_probs = pretrained_model.predict(x_init)
    x_init = tf.convert_to_tensor(x_init)
    preds = pretrained_model.predict(x_init)
    initial_class = np.argmax(preds)
    label = tf.one_hot(initial_class, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))
    g = create_adversarial_pattern(x_init, label, Mask)
    return g



