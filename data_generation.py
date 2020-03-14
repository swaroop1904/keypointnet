import tensorflow as tf
import numpy as np

def parser(serialized_example, vh=128, vw=128):
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