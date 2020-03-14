import os
import numpy as np
import tensorflow as tf
import datetime
from model import keypoint_model, orientation_model
from utils import post_process_orient, Transformer, post_process_kp
from utils import pose_loss, variance_loss, separation_loss, silhouette_loss, mvc_loss, pose_loss
from utils import Transformer
from data_generation import create_data_generator

def orientation_loss(orient, mv):
    '''
    inputs: 
           orient: (batch_size,2,2)
           mv: model view matrix (batch_size, 4, 4)
    output:
           orient_loss: (batch_size, 2)
    '''
    xp_axis = tf.tile(
        tf.constant([[[1.0, 0, 0, 1], [-1.0, 0, 0, 1]]]), [tf.shape(orient)[0], 1, 1]
        )

    xp = tf.matmul(xp_axis, mv)
    xp = t.project(xp)

    orient_loss = tf.keras.losses.MSE(orient, xp[..., :2])
    return orient_loss

def orient_net_train_step(rgb, mv):
    with tf.GradientTape() as tape:
        orient = orient_net(rgb)
        post_orient, _ = post_process_orient(orient)
        loss = orientation_loss(post_orient, mv)
    grads = tape.gradient(loss, orient_net.trainable_variables)
    optim.apply_gradients(zip(grads, orient_net.trainable_variables))
    train_orient_loss(loss)
    return orient

if __name__ == '__main__':
    vw, vh = 128, 128
    dataset_dir = '/home/swaroop/Documents/others/MS/aml/project/chairs_with_keypoints/'
    t = Transformer(vw, vh, dataset_dir)

    batch_size=64

    # remove the files other tf record from here
    filenames = [dataset_dir + val for val in os.listdir(dataset_dir) if val.endswith('tfrecord')  ]
    dataset = create_data_generator(filenames, batch_size=batch_size)

    orient_net = orientation_model()
    optim = tf.keras.optimizers.Adam(lr=1e-3)
    num_epochs = 3

    train_orient_loss = tf.keras.metrics.Mean('train_orient_loss', dtype=tf.float32)

    for epoch in range(num_epochs):
        for idx, data in enumerate(dataset):
            for i in range(2):
                rgb = data[f"img{i}"][..., :3]
                mv = data[f"mv{i}"]
                orient = orient_net_train_step(rgb, mv)
            if idx % 100000 == 0:
                print('loss_orient', train_orient_loss.result())   
                train_orient_loss.reset_states()
                orient_net.save_weights('orientation_network.h5')       
