"""
This module defines the training loop for keypointnet
"""

import os
import numpy as np
import tensorflow as tf
import datetime
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from model import keypoint_model, orientation_model
from utils import post_process_orient, Transformer, post_process_kp
from utils import pose_loss, variance_loss, separation_loss, silhouette_loss, mvc_loss, pose_loss
from utils import Transformer
from data_generation import create_data_generator
import argparse

def keypoint_loss(prob, z, images, mv_list, mvi_list, delta=0.05, num_kp=10, batch_size=2):
    """
    Calculates the loss for the keypointnet.

    Args:
        prob, z: list of the outputs of keypoint network for each of the image
                 (batch_size,  128, 128, 10)
        
    """
    sil_loss = 0.0
    var_loss = 0.0
    sep_loss = 0.0
    mv_loss = 0.0

    tot_loss = 0.0

    uvz = [None] * 2
    uvz_proj = [None] * 2  # uvz coordinates projected on to the other view.
    
    noise = 0.1
    for i in range(2):
        exp_uv, exp_z = post_process_kp(prob[i], z[i])
        sil_loss += silhouette_loss(images[i], prob[i], z[i])
        var_loss += variance_loss(exp_uv, prob[i])

        uvz[i] = tf.concat([exp_uv, exp_z], axis=2)
        world_coords = tf.matmul(t.unproject(uvz[i]), mvi_list[i])

        # [batch, num_kp, 3]
        # the projection of the object in the second image onto first
        uvz_proj[i] = t.project(tf.matmul(world_coords, mv_list[1 - i]))

    for i in range(2):
        sep_loss += separation_loss(t.unproject(uvz[i])[:, :, :3], delta, batch_size)
        mv_loss += mvc_loss(uvz_proj[i][:, :, :2], uvz[1 - i][:, :, :2])

    p_loss = pose_loss(
      tf.matmul(mvi_list[0], mv_list[1]),
      t.unproject(uvz[0])[:, :, :3],
      t.unproject(uvz[1])[:, :, :3],
      noise)

    train_sil_loss(sil_loss)
    train_var_loss(var_loss)
    train_sep_loss(sep_loss)
    train_mvc_loss(mv_loss)
    train_pose_loss(p_loss)

    tot_loss = sil_loss + 0.5*var_loss + sep_loss + mv_loss + 0.2*p_loss

    train_total_loss(tot_loss)
    return tot_loss

def keypointnet_train_step(data, batch_size):
    with tf.GradientTape() as tape:
        prob = [None, None]
        z = [None, None]
        images = [None, None]
        mv_list = [None, None]
        mvi_list = [None, None]

        for i in range(2):
            rgb = data[f"img{i}"][..., :3]
            mv = data[f"mv{i}"]
            mvi = data[f"mvi{i}"]
            orient_gt = data[f"lr{i}"]
            
            # orient net output is not utilized during training
            orient = orient_net(rgb)
            _, tiled_orientation = post_process_orient(orient, orient_gt=orient_gt)
            rgbo = tf.concat([rgb, tiled_orientation], axis=3)
            
            prob[i], z[i] = keypointnet(rgbo)
            images[i] = data[f"img{i}"]
            mv_list[i] = mv
            mvi_list[i] = mvi
        loss = keypoint_loss(prob, z, images, mv_list, mvi_list, batch_size=batch_size)
    grads = tape.gradient(loss, keypointnet.trainable_variables)
    optim.apply_gradients(zip(grads, keypointnet.trainable_variables))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./main_keypoint.py")
    parser.add_argument(
        '--dataset_dir', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--batch_size', '-bs',
        type=int,
        default=5,
        help='Batch size',
    )
    parser.add_argument(
        '--num_epochs', '-n',
        type=int,
        required=True,
        help='Batch size',
    )

    FLAGS, unparsed = parser.parse_known_args()

    dataset_dir = FLAGS.dataset_dir 
    batch_size = FLAGS.batch_size 
    num_epochs = FLAGS.num_epochs

    vw, vh = 128, 128
    t = Transformer(vw, vh, dataset_dir)

    # ignore files other than tf records 
    filenames = [dataset_dir + val for val in os.listdir(dataset_dir) if val.endswith('tfrecord')  ]
    dataset = create_data_generator(filenames, batch_size=batch_size)

    orient_net = orientation_model()
    optim = tf.keras.optimizers.Adam(lr=1e-3)
    
    keypointnet = keypoint_model()
    
    train_sil_loss = tf.keras.metrics.Mean('train_sil_loss', dtype=tf.float32)
    train_var_loss = tf.keras.metrics.Mean('train_var_loss', dtype=tf.float32)
    train_mvc_loss = tf.keras.metrics.Mean('train_mvc_loss', dtype=tf.float32)
    train_sep_loss = tf.keras.metrics.Mean('train_sep_loss', dtype=tf.float32)
    train_pose_loss = tf.keras.metrics.Mean('train_pose_loss', dtype=tf.float32)
    train_total_loss = tf.keras.metrics.Mean('train_total_loss', dtype=tf.float32)

    for epoch in range(num_epochs):
        for idx, data in enumerate(dataset):
            keypointnet_train_step(data, batch_size)
            if idx % 100000 == 0:
                print('sil_loss', train_sil_loss.result())
                print('var', train_var_loss.result())
                print('mvc loss', train_mvc_loss.result())
                print('sep loss', train_sep_loss.result())
                print('pose', train_pose_loss.result())
                print('train', train_total_loss.result())
                train_sil_loss.reset_states()
                train_var_loss.reset_states()
                train_mvc_loss.reset_states()
                train_sep_loss.reset_states()
                train_pose_loss.reset_states()
                train_total_loss.reset_states()
                keypointnet.save_weights('keypoint_network.h5')
                
        print("**"*40)
        print("completed epoch")
        print("**"*40)