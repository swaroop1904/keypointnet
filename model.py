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

def create_orientation_model(vh=128, vw=128):
    '''
    creates orientation networks with 32 filters per layer
    output: batch_size * 2 * 2 
    The output is defined in normalized camera coordinates
    '''
    inp, bb_op = get_dilated_backbone(vh, vw, 3, 32)
    prob = Conv2D(2, 3, dilation_rate=1)(bb_op)

    model = Model(inp, prob)
    return model

def create_keypoint_model(vh=128, vw=128, num_kp = 10):
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

def mesh_grid(h):
    '''
    creates a mesh grid with normalized pixel values
    '''
    r = np.arange(0.5, h, 1) / (h / 2) - 1
    xx, yy = tf.meshgrid(r, -r)
    return tf.to_float(xx), tf.to_float(yy)

def post_process_kp(prob, z):
    '''
    expected value of uv and z
    uv is in normalized image coordinates systen
    '''
    prob = prob[0,3,1,2]
    z = z[0,3,1,2]

    prob = prob.reshape([-1, num_kp, vh*vw])

    prob = Activation('softmax')(prob)
    prob = prob.reshape([-1, num_kp, vh, vw])

    # haven't added the visualization code
    
    xx, yy = mesh_grid(vh)

    sx = tf.reduce_sum(prob*xx, axis=[2,3])
    sy = tf.reduce_sum(prob*yy, axis=[2,3])

    z = tf.reduce_sum(prob*z, axis=[2,3])
    uv = tf.reshape(tf.stack([sx,sy], -1), [-1, num_kp, 2])

    return uv, z

def post_process_orient(orient, vh=128, vw=128, anneal=1, orient_gt=None):
    '''
    calculates the orientation of the orient network
    it is equal to orient_gt for train and orient_pred for test
    it is tiled to match the shape of the images
    '''
    orient_pred = tf.maximum(0.0, tf.sign(orient[:,0,:1] - orient[:,1,:1]))

    if orient_gt:
        orient_gt = tf.maximum(0, tf.sign(orient_gt[:,:,1]))
        orient = tf.round(orient_gt*anneal + orient_pred *(1-anneal)) 
    else:
        orient = orient_pred

    orient_tiled = tf.tile(tf.expand_dims(tf.expand_dims(orient, 1), 1),
                           [1, vh, vw, 1]) 

    return orient_tiled

def estimate_rotation(obj0, obj1, noise=0.1):
    '''
    estimates the rotation given coordinates of the keypoints of two views
    '''

    obj0 += tf.random_normal(tf.shape(obj0), mean=0, stddev=noise)
    obj1 += tf.random_normal(tf.shape(obj1), mean=0, stddev=noise)

    mean0 = tf.reduce_mean(obj0, 1, keepdims=True)
    mean1 = tf.reduce_mean(obj1, 1, keepdims=True)

    obj0 = obj0 - mean0
    obj1 = obj1 - mean1
    
    cov = tf.matmul(tf.transpose(obj0), obj1)
    _, u, v = tf.linalg.svd(cov, full_matrices=True)

    det = tf.matrix_determinant(tf.matmul(v,tf.transpose(u)))

    ud = tf.concat(
        [u[:,:,:-1], u[:,:,-1:] * tf.expand_dims(tf.expand_dims(d,1),1)],
        axis=2)

    return tf.matmul(ud, v, transpose_b=True)

def pose_loss(gt_homogeneous, obj0, obj1, noise):
    estimated_rot_t = estimate_rotation(obj0, obj1, noise)
    gt_rotation = gt_homogeneous[:, :3, :3]
    frob = tf.sqrt(tf.reduce_sum(tf.square(estimated_rot_t - gt_rotation), axis=[1,2]))

    return tf.reduce_mean(tf.square(frob)), \
            2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0, frob / 2 * math.sqrt(2))))


def mvc_loss(uv0, uv1):
    '''
    calculate the keypoint location difference between one pair of 
    projected and predicted keypoints. 
    '''

    diff = tf.reduce_sum(tf.square(uv0-uv1), axis=[1,2])
    return tf.reduce_mean(diff)
    

def silhouette_loss(input_img, prob, z, vh=128, vw=128):
    '''
    '''
    uv, _ = post_process_kp(prob, z)

    mask = input_img[..., 3]
    mask = tf.cast(tf.greater(mask, tf.zeros_like(mask)), dtype=tf.float32)

    prob = prob[0,3,1,2]
    prob = prob.reshape([-1, num_kp, vh*vw])

    prob = Activation('softmax')(prob)
    prob = prob.reshape([-1, num_kp, vh, vw])

    sill = tf.reduce_sum(prob * tf.expand_dims(mask, 1), axis=[2,3])
    sill = tf.reduce_mean(-tf.log(sill + 1e-12))

    v_loss = variance_loss(uv, prob, vh)

    return sill 


def variance_loss(uv, prob, vh):
    '''
    uv is in ndc
    '''    
    xx, yy = mesh_grid(vh)

    xy = tf.stack([xx, yy], axis=2)
    sh = tf.shape(xy)

    xy = tf.reshape(xy, [1,1, sh[0], sh[1], 2])
    sh = tf.shape(uv)

    uv = tf.reshape(uv, [sh[0], sh[1], 1, 1, 2])
    diff = tf.reduce_sum(tf.square(uv-xy), axis=4)
    diff *= prob

    return tf.reduce_mean(tf.reduce_sum(diff, axis=[2,3]))


def separation_loss(xyz, delta, batch_size):
    num_kp = tf.shape(xyz)[1]

    t1 = tf.tile(xyz, [1,num_kp, 1])
    t2 = tf.reshape(tf.tile(xyz, [1,1,num_kp]), tf.shape(t1))

    diff_sq = tf.square(t1-t2)
    lensqr = tf.reduce_sum(diff_sq, axis=2)

    return tf.reduce_sum(tf.maximum(-lensqr+delta, 0.0)) / tf.to_float(num_kp * batch_size * 2)

