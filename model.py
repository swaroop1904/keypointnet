import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras import Model
import math

def get_dilated_backbone(vh, vw, inp_c, num_filters):
    '''
    creates backbone architecture as defined in keypointnet paper
    '''
    inp = Input(shape=(vh, vw, inp_c))
    x = inp

    dilation_rates = [1,1,2,4,8,16,1,2,4,8,16,1]

    for dr in dilation_rates:
        x = Conv2D(num_filters, 3, padding='same', dilation_rate=dr)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return inp, x

def orientation_model(vh=128, vw=128):
    '''
    creates orientation networks with 32 filters per layer
    output: batch_size * 2 * 2 
    The output is defined in normalized camera coordinates
    '''
    inp, bb_op = get_dilated_backbone(vh, vw, 3, 32)
    prob = Conv2D(2, 3, dilation_rate=1)(bb_op)

    model = Model(inp, prob)
    return model

def keypoint_model(vh=128, vw=128, num_kp = 10):
    '''
    creates keypoint network with 64 filters per layers
    output: 
            prob: batchsize * vh * vw * num_kp 
            z: batchsize * vh * vw * num_kp
    '''
    inp, bb_op = get_dilated_backbone(vh, vw, 4, 64)

    prob = Conv2D(num_kp, 3, padding='same', dilation_rate=1)(bb_op)
    z = -30 + Conv2D(num_kp, 3, padding='same', dilation_rate=1)(bb_op)

    model = Model(inputs=inp, outputs=[prob, z])
    return model