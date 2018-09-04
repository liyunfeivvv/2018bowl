import os
import glob
import numpy as np
import tensorflow as tf
import random
import cv2


seed = 1234
random.seed = seed
np.random.seed = seed

# Set some parameters
IMG_WIDTH = 240
IMG_HEIGHT = 240
IMG_CHANNELS = 3
TRAIN_PATH = './stage1_train/'


def load_sample(root_dir, name):
    img_path = os.path.join(root_dir, name, 'images', name + '.png')
    mask_path = glob.glob(os.path.join(root_dir, name, 'masks/') + '*.png')

    img_BGR = cv2.imread(img_path)
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    img_YCrCb = cv2.resize(img_YCrCb, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

    # if white convert to black
    if np.mean(img_YCrCb[..., 0]) > 127.:
        img_YCrCb[..., 0] = 255 - img_YCrCb[..., 0]

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_path_i in mask_path:
        mask_i = cv2.imread(mask_path_i)
        mask_i = cv2.cvtColor(mask_i, cv2.COLOR_BGR2GRAY)
        mask_i = cv2.resize(mask_i, (IMG_HEIGHT, IMG_WIDTH))
        mask_i = np.expand_dims(mask_i, axis=-1)
        mask = np.maximum(mask, mask_i)

    mask = mask.astype(np.uint8)
    img_RGB = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
    return img_RGB, mask


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def convert_to_example(img, mask):
    image_raw = img.tostring()
    image_shape = [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
    image_format = 'RAW'

    mask_raw = mask.tostring()
    mask_shape = [IMG_HEIGHT, IMG_WIDTH, 1]
    mask_format = 'RAW'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/format': _bytes_feature(image_format.encode()),
        'image/shape': _int64_feature(image_shape),
        'image/encoded': _bytes_feature(image_raw),
        'mask/format': _bytes_feature(mask_format.encode()),
        'mask/shape': _int64_feature(mask_shape),
        'mask/encoded': _bytes_feature(mask_raw),
    }))

    return example


def convert_to_tfrecords(root_dir, save_dir, num_shards=4):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_ids = next(os.walk(root_dir))[1]

    spacing = np.linspace(0., len(all_ids), num_shards + 1).astype(int)

    print('--- start ---')

    for shard in range(num_shards):

        print("Shard:[{0}/{1}]".format(shard, num_shards))
        saved_tfrecord_path = '{0}{1}train.tfrecords-{2:0>3d}of-{3:0>3d}'.format(save_dir, IMG_HEIGHT, shard, num_shards)

        with tf.python_io.TFRecordWriter(saved_tfrecord_path) as writer:
            now_ids = all_ids[spacing[shard]: spacing[shard + 1]]

            for id_ in now_ids:
                img, mask = load_sample(root_dir, id_)
                example = convert_to_example(img, mask)
                writer.write(example.SerializeToString())

    print('--- end ---')


if __name__ == '__main__':
    convert_to_tfrecords(TRAIN_PATH, './tfrecords/train/')
