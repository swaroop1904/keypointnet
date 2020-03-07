from model import keypoint_model, orientation_model
from utils import post_process_orient, Transformer, post_process_kp
from utils import pose_loss, variance_loss, separation_loss, silhouette_loss, mvc_loss, pose_loss
from utils import Transformer
import os
import numpy as np
import tensorflow as tf
from IPython.core.debugger import set_trace

def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    fs = tf.io.parse_single_example(
        serialized_example,
        features={
            "img0": tf.io.FixedLenFeature([], tf.string),
            "img1": tf.io.FixedLenFeature([], tf.string),
            "mv0": tf.io.FixedLenFeature([16], tf.float32),
            "mvi0": tf.io.FixedLenFeature([16], tf.float32),
            "mv1": tf.io.FixedLenFeature([16], tf.float32),
            "mvi1": tf.io.FixedLenFeature([16], tf.float32),
        })

    fs["img0"] = tf.math.divide(tf.cast(tf.image.decode_png(fs["img0"], 4), tf.float32), 255)
    fs["img1"] = tf.math.divide(tf.cast(tf.image.decode_png(fs["img1"], 4), tf.float32), 255)

    fs["img0"].set_shape([vh, vw, 4])
    fs["img1"].set_shape([vh, vw, 4])


    fs["mv1"] = tf.transpose(tf.reshape(fs["mv1"], [4, 4]), [1,0])
    fs["mvi1"] = tf.transpose(tf.reshape(fs["mvi1"], [4, 4]), [1, 0])
    fs["mv0"] = tf.transpose(tf.reshape(fs["mv0"], [4, 4]), [1, 0])
    fs["mvi0"] = tf.transpose(tf.reshape(fs["mvi0"], [4, 4]), [1, 0])

    fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
    fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])

    return fs

def create_data_generator(filenames, batch_size):
    np.random.shuffle(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser, num_parallel_calls=4)
    dataset = dataset.shuffle(400).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=50)

    return dataset

def orientation_loss(orient, mv):
    xp_axis = tf.tile(
        tf.constant([[[1.0, 0, 0, 1], [-1.0, 0, 0, 1]]]), [tf.shape(orient)[0], 1, 1]
        )

    xp = tf.matmul(xp_axis, mv)
    xp = t.project(xp)

    orient_loss = tf.reduce_mean(tf.keras.losses.MSE(orient[...,:2], xp[..., :2]))
    return orient_loss

def orient_net_train_step(rgb, mv):
    with tf.GradientTape() as tape:
        orient = orient_net(rgb)
        post_orient, _ = post_process_orient(orient)
        loss = orientation_loss(post_orient, mv)
    grads = tape.gradient(loss, orient_net.trainable_variables)
    optim.apply_gradients(zip(grads, orient_net.trainable_variables))

    return orient


def keypoint_loss(prob, z, images, mv_list, mvi_list, delta=0.05, num_kp=10, batch_size=2):
    sil_loss = 0.0
    var_loss = 0.0
    sep_loss = 0.0
    mv_loss = 0.0

    tot_loss = 0.0

    uvz = [None] * 2
    uvz_proj = [None] * 2  # uvz coordinates projected on to the other view.
    
    noise = 0.1

    for i in range(2):
        #set_trace()
        exp_uv, exp_z = post_process_kp(prob[i], z[i])
        sil_loss += silhouette_loss(images[i], prob[i], z[i])
        var_loss += variance_loss(exp_uv, prob[i])

        exp_uv = tf.reshape(exp_uv, [-1, num_kp, 2])
        exp_z = tf.reshape(exp_z, [-1, num_kp, 1])

        uvz[i] = tf.concat([exp_uv, exp_z], axis=2)

        world_coords = tf.matmul(t.unproject(uvz[i]), mvi_list[i])

        # [batch, num_kp, 3]
        uvz_proj[i] = t.project(tf.matmul(world_coords, mv_list[1 - i]))

    for i in range(2):
        sep_loss += separation_loss(t.unproject(uvz[i])[:, :, :3], delta, batch_size)
        mv_loss += mvc_loss(uvz_proj[i][:, :, :2], uvz[1 - i][:, :, :2])

    p_loss = pose_loss(
      tf.matmul(mvi_list[0], mv_list[1]),
      t.unproject(uvz[0])[:, :, :3],
      t.unproject(uvz[1])[:, :, :3],
      noise)


    tot_loss = sil_loss + var_loss + sep_loss + mv_loss + p_loss
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
            images[i] = data[f"img{i}"]
            mv_list[i] = mv
            mvi_list[i] = mvi
            orient = orient_net(rgb)
            _, tiled_orientation = post_process_orient(orient)
            rgbo = tf.concat([rgb, tiled_orientation], axis=3)
            prob[i], z[i] = keypointnet(rgbo)
        loss = keypoint_loss(prob, z, images, mv_list, mvi_list, batch_size)
    grads = tape.gradient(loss, keypointnet.trainable_variables)
    optim.apply_gradients(zip(grads, keypointnet.trainable_variables))

if __name__ == '__main__':
    vw, vh = 128, 128
    dataset_dir = '/home/swaroop/Documents/others/MS/aml/project/chairs_with_keypoints/'
    t = Transformer(vw, vh, dataset_dir)

    batch_size=2

    # remove the files other tf record from here
    filenames = [dataset_dir + val for val in os.listdir(dataset_dir) if val.endswith('tfrecord')  ]
    dataset = create_data_generator(filenames, batch_size=batch_size)

    orient_net = orientation_model()
    keypointnet = keypoint_model()

    optim = tf.keras.optimizers.Adam(lr=1e-3)
    num_epochs = 2

    for epoch in range(num_epochs):
        for data in dataset:
            for i in range(2):
                rgb = data[f"img{i}"][..., :3]
                mv = data[f"mv{i}"]
                orient = orient_net_train_step(rgb, mv)
    
    orient_net.save_weights('orientation_network.h5')

    for epoch in range(num_epochs):
        for data in dataset:
            keypointnet_train_step(data, batch_size)

    keypoint_net.save_weights('keypoint_network.h5')






        
