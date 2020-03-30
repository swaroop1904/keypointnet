"""
This module implements the utility functions and loss functions required for
keypoint implementation. The methods defined below have been adapted from
or directly taken from the repo: 
https://github.com/tensorflow/models/tree/master/research/keypointnet 

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation 
import math
import os
import numpy as np
from copy import deepcopy

def mesh_grid(h):
    """
    Creates a mesh grid with normalized pixel values.
    
    """
    r = np.arange(0.5, h, 1) / (h / 2) - 1
    xx, yy = tf.meshgrid(r, -r)
    return tf.cast(xx, tf.float32), tf.cast(yy, tf.float32)

# when testing need to set anneal to 0 
def post_process_orient(orient_prob, anneal=1, orient_gt=None, vh=128, vw=128):
    """
    Post process the orient network output

    
    Args:
          orient_prob: (batch_size, 128, 128, 2)
          anneal: 1 -> use gt orientation (used during training)
                  0 -> use orient_pred (used during inference)
          orient_gt:   camera co-ordinates of NDC points (1,0,0) and (-1,0,0)
                       (batch_size, 2, 2)
    Returns:
           out_xy:       predicted camera co-ordinates of NDC points (1,0,0) and
                         (-1,0,0)
                         (batch_size, 2, 2)
           orient_tiled: Tiled tensor for the orientation to be fed along with image
                         to the keypoint network
                         (batch_size, 128, 128, 1)
    """

    orient_prob = tf.transpose(orient_prob, [0,3,1,2])
    orient_prob = tf.reshape(orient_prob, [-1, 2, vh*vw])
    orient_prob = tf.nn.softmax(orient_prob)
    orient_prob = tf.reshape(orient_prob, [-1, 2, vh, vw])

    xx, yy = mesh_grid(vh)
    
    sx = tf.reduce_sum(orient_prob*xx, axis=[2,3])
    sy = tf.reduce_sum(orient_prob*yy, axis=[2,3])

    # actual output of orient network
    out_xy = tf.reshape(tf.stack([sx, sy], -1), [-1, 2, 2])

    # orientation of the output. Calculated to be given as i/p to keypointnet
    orient_pred = tf.maximum(0.0, tf.sign(out_xy[:,0,:1] - out_xy[:,1,:1]))

    if orient_gt is None:
        orient = orient_pred
    else:
        orient_gt = tf.maximum(0, tf.sign(orient_gt[...,1]))
        orient = tf.round(orient_gt*anneal + orient_pred *(1-anneal)) 

    # tile orientation to be used as one of the channels for keypointnet inp
    orient_tiled = tf.tile(tf.expand_dims(tf.expand_dims(orient, 1), 1),
                           [1, vh, vw, 1]) 

    return out_xy, orient_tiled


def post_process_kp(prob, z, num_kp=10, vh=128, vw=128):
    """
    Calculates the expected value of uv and z.

    Args:
        prob, z: output of keypoint network a image
                (batch_size,  128, 128, 10)
    Returns:
        uv: expected value of u,v location in image coordinates system
        z: expected value of z 
    """


    z = tf.transpose(z, [0,3,1,2])
    prob = get_probability_distribution(prob)

    xx, yy = mesh_grid(vh)

    sx = tf.reduce_sum(prob*xx, axis=[2,3])
    sy = tf.reduce_sum(prob*yy, axis=[2,3])

    z = tf.reduce_sum(prob*z, axis=[2,3])
    uv = tf.reshape(tf.stack([sx,sy], -1), [-1, num_kp, 2])
    z = tf.reshape(z, [-1, num_kp, 1])

    return uv, z

def estimate_rotation(obj0, obj1, noise=0.1):
    """
    Estimates the rotation given coordinates of the keypoints of two views.

    Uses proscutes algorithm to determine the orientation between the two
    objects. 

    Args:
            obj0: first set of 3D keypoints 
                  (batch_size, num_kp, 3)
            obj1: second set of 3D keypoints
                  (batch_size, num_kp, 3)
            noise: scalar noise added to keypoints
    """

    obj0 += tf.random.normal(tf.shape(obj0), mean=0, stddev=noise)
    obj1 += tf.random.normal(tf.shape(obj1), mean=0, stddev=noise)

    mean0 = tf.reduce_mean(obj0, 1, keepdims=True)
    mean1 = tf.reduce_mean(obj1, 1, keepdims=True)

    obj0 = obj0 - mean0
    obj1 = obj1 - mean1
    
    cov = tf.matmul(tf.transpose(obj0, [0,2,1]), obj1)

    _, u, v = tf.linalg.svd(cov, full_matrices=True)

    det = tf.linalg.det(tf.matmul(v,tf.transpose(u, [0,2,1])))

    ud = tf.concat(
        [u[:,:,:-1], u[:,:,-1:] * tf.expand_dims(tf.expand_dims(det,1),1)],
        axis=2)

    return tf.matmul(ud, v, transpose_b=True)

def pose_loss(gt_homogeneous, obj0, obj1, noise):
    """
    Args:
            obj0: first set of 3D keypoints 
                  (batch_size, num_kp, 3)
            obj1: second set of 3D keypoints
                  (batch_size, num_kp, 3)
            gt_homogeneous: true rotation matrix
                  (batch_size, 4, 4)
            noise: scalar noise added to keypoints
    """
    estimated_rot_t = estimate_rotation(obj0, obj1, noise)

    gt_rotation = gt_homogeneous[:, :3, :3]
    frob = tf.sqrt(tf.reduce_sum(tf.square(estimated_rot_t - gt_rotation), axis=[1,2]))

    return 2.0 * tf.reduce_mean(tf.asin(tf.minimum(1.0, frob / (2 * math.sqrt(2)))))


def mvc_loss(uv0, uv1):
    """
    Args: 
            uv1: predicted 2D keypoint of the first image
                 (batch_size, num_kp, 2)
            uv0: projected 2D keypoints from the other image
                 (batch_size, num_kp, 2)
    """

    diff = tf.reduce_sum(tf.square(uv0-uv1), axis=[1,2])
    return tf.reduce_mean(diff)
    

def silhouette_loss(input_img, prob, z, vh=128, vw=128, num_kp=10):
    """
    Args:
        prob, z: Output of keypoint network for a single image
                (batch_size,  128, 128, 10)
    """
    mask = input_img[..., 3]
    mask = tf.cast(tf.greater(mask, tf.zeros_like(mask)), dtype=tf.float32)

    prob = get_probability_distribution(prob)

    sill = tf.reduce_sum(prob * tf.expand_dims(mask, 1), axis=[2,3])
    sill = tf.reduce_mean(-tf.math.log(sill + 1e-12))

    return sill

def get_probability_distribution(prob, num_kp=10, vh=128, vw=128):
    """
    Converts the keypointnet output feature maps into prob distribution
    by applying spatial softmax.

    """
    prob = tf.transpose(prob, [0,3,1,2])
    prob = tf.reshape(prob, [-1, num_kp, vh*vw])

    prob = Activation('softmax')(prob)
    prob = tf.reshape(prob, [-1, num_kp, vh, vw])

    return prob

def variance_loss(uv, prob, vh=128, vw=128, num_kp=10):
    """

    Args:
          uv is in ndc: (batch_size, 10, 2)
          prob is the keypointnet output (batch_size, 128, 128, 10)
    """    
    prob = get_probability_distribution(prob)
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
    """
    Args:
           xyz: keypoints in camera co-ordinates (batch_size, num_kp, 3)
           delta: separation threshold. Incurs 0 cost if separation is greater
                  than delta  
    """
    num_kp = tf.shape(xyz)[1]

    t1 = tf.tile(xyz, [1,num_kp, 1])
    t2 = tf.reshape(tf.tile(xyz, [1,1,num_kp]), tf.shape(t1))

    diff_sq = tf.square(t1-t2)
    lensqr = tf.reduce_sum(diff_sq, axis=2)

    return tf.reduce_sum(tf.maximum(-lensqr+delta, 0.0)) / tf.cast(num_kp * batch_size * 2, tf.float32)


class Transformer(object):
  """A utility for projecting 3D points to 2D coordinates and vice versa.

  3D points are represented in 4D-homogeneous world coordinates. The pixel
  coordinates are represented in normalized device coordinates [-1, 1].
  See https://learnopengl.com/Getting-started/Coordinate-Systems.
  """

  def __get_matrix(self, lines):
    return np.array([[float(y) for y in x.strip().split(" ")] for x in lines])

  def __read_projection_matrix(self, filename):
    with open(filename, "r") as f:
      lines = f.readlines()
    return self.__get_matrix(lines)

  def __init__(self, w, h, dataset_dir):
    self.w = w
    self.h = h
    p = self.__read_projection_matrix(dataset_dir + "projection.txt")

    # transposed of inversed projection matrix.
    self.pinv_t = tf.constant([[1.0 / p[0, 0], 0, 0,
                                0], [0, 1.0 / p[1, 1], 0, 0], [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=tf.float32)
    self.f = p[0, 0]

  def project(self, xyzw):
    """Projects homogeneous 3D coordinates to normalized device coordinates."""

    z = xyzw[:, :, 2:3] + 1e-8
    return tf.concat([-self.f * xyzw[:, :, :2] / z, z], axis=2)

  def unproject(self, xyz):
    """Unprojects normalized device coordinates with depth to 3D coordinates."""

    z = xyz[:, :, 2:]
    xy = -xyz * z

    def batch_matmul(a, b):
      return tf.reshape(
          tf.matmul(tf.reshape(a, [-1, a.shape[2]]), b),
          [-1, a.shape[1], a.shape[2]])

    return batch_matmul(
        tf.concat([xy[:, :, :2], z, tf.ones_like(z, dtype=tf.float32)], axis=2), self.pinv_t)

