"""
This module includes methods for creation of keypointnet and orientnet

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras import Model
import math

def get_dilated_backbone(vh, vw, inp_c, num_filters):
    """
    Creates a 12 layer backbone CNN model with varied dilation rates.
    
    Returns: 
            (batch_size, 128, 128, num_filters)
    """
    inp = Input(shape=(vh, vw, inp_c))
    x = inp

    dilation_rates = [1,1,2,4,8,16,1,2,4,8,16,1]

    for dr in dilation_rates:
        x = Conv2D(num_filters, 3, padding='same', dilation_rate=dr)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return inp, x

def orientation_model(vh=128, vw=128):
    """
    Creates orientation networks with 32 filters per layer.

    Returns: 
            batch_size * 128 * 128 * 2
    """
    inp, bb_op = get_dilated_backbone(vh, vw, 3, 32)
    prob = Conv2D(2, 3, dilation_rate=1, padding='same')(bb_op)

    model = Model(inp, prob)
    return model

def keypoint_model(vh=128, vw=128, num_kp = 10):
    """
    Creates keypoint network with 64 filters per layers.
    
    Returns: 
            prob: batchsize * vh * vw * num_kp 
            z: batchsize * vh * vw * num_kp
    """
    inp, bb_op = get_dilated_backbone(vh, vw, 4, 64)

    prob = Conv2D(num_kp, 3, padding='same', dilation_rate=1)(bb_op)
    z = -30 + Conv2D(num_kp, 3, padding='same', dilation_rate=1)(bb_op)

    model = Model(inputs=inp, outputs=[prob, z])
    return model