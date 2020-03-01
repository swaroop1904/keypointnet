import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras import Model


def get_dilated_backbone(vh, vw, num_filters):
    inp = Input(shape=(vh, vw, 4))
    x = inp

    dilation_rates = [1,1,2,4,8,16,1,2,4,8,16,1,1]

    for dr in dilation_rates:
        x = Conv2D(num_filters, 3, padding='same', dilation_rate=dr)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)

    return inp, x

def mesh_grid(h):
    r = np.arange(0.5, h, 1) / (h / 2) - 1
    xx, yy = tf.meshgrid(r, -r)
    return tf.to_float(xx), tf.to_float(yy)

def create_orientation_model():
    inp, bb_op = get_dilated_backbone(vh, vw, 64)
    prob = Conv2D(2, 3, dilation_rate=1)(bb_op)
    model = Model(inp, prob)
    return model


def create_keypoint_model():
    vh, vw = 128, 128
    num_kp = 10
    inp, bb_op = get_dilated_backbone(vh, vw, 64)

    prob = Conv2D(num_kp, 3, padding='same', dilation_rate=1)(bb_op)
    z = -30 + Conv2D(num_kp, 3, padding='same', dilation_rate=1)(bb_op)

    model = Model(inputs=inp, outputs=[prob, z])
    return model

def post_process(prob, z):
    prob = prob[0,3,1,2]
    z = z[0,3,1,2]

    prob = prob.reshape([-1, num_kp, vh*vw])
    z = z.reshape([-1, num_kp, vh*vw])

    prob = Activation('Sigmoid')(prob)
    prob = prob.reshape([-1, num_kp, vh, vw])

    # haven't added the visualization code
    
    xx, yy = mesh_grid(vh)

    sx = tf.reduce_sum(prob*xx, axis=[2,3])
    sy = tf.reduce_sum(prob*yy, axis=[2,3])

    z = tf.reduce_sum(prob*z, axis=[2,3])
    uv = tf.reshape(tf.stack([sx,sy], -1), [-1, num_kp, 2])

    return uv, z

def estimate_rotation(obj0, obj1):
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

def pose_loss(gt_homogeneous, obj0, obj1):
    estimated_rot = estimate_rotation(obj0, obj1)
    gt_rotation = gt_homogeneous[:, :3, :3]
    frob = tf.sqrt(tf.reduce_sum(tf.square(estimate_rot - gt_rotation), axis=[1,2]))

    return tf.reduce_mean(tf.square(frob)), \
            2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0, frob / 2 * math.sqrt(2))))


def mvc_loss():


def silhouette_loss():


def separation_loss():