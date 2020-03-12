from model import keypoint_model, orientation_model
from utils import post_process_orient, Transformer, post_process_kp
from utils import pose_loss, variance_loss, separation_loss, silhouette_loss, mvc_loss, pose_loss
from utils import Transformer
import os
import numpy as np
import tensorflow as tf
import datetime
#from IPython.core.debugger import set_trace

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

    batch_size=2

    # remove the files other tf record from here
    filenames = [dataset_dir + val for val in os.listdir(dataset_dir) if val.endswith('tfrecord')  ]
    dataset = create_data_generator(filenames, batch_size=batch_size)

    orient_net = orientation_model()
    optim = tf.keras.optimizers.Adam(lr=1e-3)
    num_epochs = 1

    train_orient_loss = tf.keras.metrics.Mean('train_orient_loss', dtype=tf.float32)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/orient/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for epoch in range(num_epochs):
        for idx, data in enumerate(dataset):
            for i in range(2):
                rgb = data[f"img{i}"][..., :3]
                mv = data[f"mv{i}"]
                orient = orient_net_train_step(rgb, mv)
            if idx % 50000 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss_orient', train_orient_loss.result(), step=int(idx/50000))   
                train_orient_loss.reset_states()
                orient_net.save_weights('orientation_network.h5')       
