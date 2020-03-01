from model import keypoint_model, orientation_model
from utils import post_process_orient, Transformer, post_process_kp
from utils import pose_loss, variance_loss, separation_loss, silhouette_loss, mvc_loss, pose_loss
from utils import Transformer

def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    fs = tf.parse_single_example(
        serialized_example,
        features={
            "img0": tf.FixedLenFeature([], tf.string),
            "img1": tf.FixedLenFeature([], tf.string),
            "mv0": tf.FixedLenFeature([16], tf.float32),
            "mvi0": tf.FixedLenFeature([16], tf.float32),
            "mv1": tf.FixedLenFeature([16], tf.float32),
            "mvi1": tf.FixedLenFeature([16], tf.float32),
        })

    fs["img0"] = tf.div(tf.to_float(tf.image.decode_png(fs["img0"], 4)), 255)
    fs["img1"] = tf.div(tf.to_float(tf.image.decode_png(fs["img1"], 4)), 255)

    fs["img0"].set_shape([vh, vw, 4])
    fs["img1"].set_shape([vh, vw, 4])

    # fs["lr0"] = [fs["mv0"][0]]
    # fs["lr1"] = [fs["mv1"][0]]

    fs["lr0"] = tf.convert_to_tensor([fs["mv0"][0]])
    fs["lr1"] = tf.convert_to_tensor([fs["mv1"][0]])

    return fs

def create_data_generator(filenames):
    np.random.shuffle(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser, num_parallel_calls=4)
    dataset = dataset.shuffle(400).repeat().batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)

    return dataset

def orientation_loss(orient, mv):
    xp_axis = tf.tile(
        tf.constant([[[1.0, 0, 0, 1], [-1.0, 0, 0, 1]]], [tf.shape(orient[0], 1, 1)])
    )
    xp = tf.matmul(xp_axis, mv)
    xp = t.project(xp)
    orient_loss = tf.losses.mean_squared_error(orient[...,:2], xp[..., :2])
    return orient_loss

def orient_net_train_step(rgb, mv):
    with tf.GradientTape as tape:
        orient = orient_net(rgb)
        loss = orientation_loss(orient, mv)
    grads = tape.gradient(loss, orient_net.trainable_variables)
    optim.apply_gradients(zip(grads, orient_net.trainable_variables))

    return orient


def keypoint_loss(prob, z, images, mv_list, mvi_list, delta=0.05):
    sil_loss = 0.0
    var_loss = 0.0
    sep_loss = 0.0
    mv_loss = 0.0
    pose_loss = 0.0

    tot_loss = 0.0

    uvz = [None] * 2
    uvz_proj = [None] * 2  # uvz coordinates projected on to the other view.
    
    for i in range(2):
        uv, z = post_process_kp(prob[i], z[i])
        sil_loss += silhouette_loss(images[i], prob[i], z[i])
        var_loss += variance_loss(uv, z)[0], 
                      prob[i])

        uvz[i] = tf.concat([uv, z], axis=2)

        world_coords = tf.matmul(t.unproject(uvz[i]), mvi_list[i])

        # [batch, num_kp, 3]
        uvz_proj[i] = t.project(tf.matmul(world_coords, mv_list[1 - i]))

    for i in range(2):
        sep_loss += separation_loss(t.unproject(uvz[i])[:, :, :3], delta))
        mv_loss += mvc_loss(uvz_proj[i][:, :, :2], uvz[1 - i][:, :, :2])

    pose_loss = pose_loss(
      t.unproject(uvz[0])[:, :, :3],
      t.unproject(uvz[1])[:, :, :3], tf.matmul(mvi[0], mv[1]), pconf,
      hparams.noise)


    tot_loss = sil_loss + var_loss + sep_loss + mv_loss + pose_loss

def keypointnet_train_step(data):
    with tf.GradientTape as tape:
        prob = [None, None]
        z = [None, None]
        images = [None, None]
        mv_list = [None, None]
        mvi_list = [None, None]

        for i in range(2):
            rgb = data[f"img{i}"][..., :3]
            mv = data[f"mv{i}"]
            mvi = data[f"mvi{i}"]
            images[0] = rgb
            mv_list[0] = mv
            mvi_list[0] = mvi
            orient = orient_net(rgb, mv)
            tiled_orientation = post_process_orient(orient)
            rgbo = tf.stack([rgb, tiled_orientation], axis=3)
            prob[i], z[i] = keypointnet(rgbo)   
        loss = keypoint_loss(prob, z, images, mv_list, mvi_list)
    grads = tape.gradient(loss, keypoint_model.trainable_variables)
    optim.apply_gradients(zip(grads, keypoint_model.trainable_variables))

if __name__ == '__main__':
    dataset_dir = ''
    t = Transformer(vw, vh, dataset_dir)

    # remove the files other tf record from here
    filenames = os.listdir(dataset_dir)
    dataset = create_data_generator(filenames)

    orient_net = orientation_model()
    keypointnet = keypoint_model()

    optim = tf.keras.optimizers.Adam(lr=1e-3)

    for epoch in range(num_epochs):
        for data in dataset:
            for i in range(2):
                rgb = data[f"img{i}"][..., :3]
                mv = data[f"mv{i}"]
                orient = orient_net_train_step(rgb, mv)
    
    orient_net.save_weights('orientation_network.h5')

    for epoch in range(num_epochs):
        for data in dataset:
            keypointnet_train_step(data)

    keypoint_net.save_weights('keypoint_network.h5')






        
