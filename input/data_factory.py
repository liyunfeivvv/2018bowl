import os
import cv2
import numpy as np
import tensorflow as tf

try:
    from input.aug_utils import iaa_image_op, elastic_transform_op, random_crop_op,\
                                random_flip_op, random_rotate_op, random_noise_op, distort_color_op
except:
    from aug_utils import iaa_image_op, elastic_transform_op, random_crop_op,\
                                random_flip_op, random_rotate_op, random_noise_op, distort_color_op
slim = tf.contrib.slim

import glob
'''
todo:
    - [?] random noise;
    - [x] rotation;
    - [x] random crop;
    - [ ] re-maping gray image to random color image;
    - [ ] blur;
    - [ ] elastic deformation;
    - [ ] channel shuffle;

'''


# todo add to a dict
ORG_SIZE = 240
OUT_IMG_SIZE = 268
OUT_MASK_SIZE = 180
PAD_SIZE = (OUT_IMG_SIZE - OUT_MASK_SIZE) // 2

# ================================================================= #
#                          get train data                           #
# ================================================================= #
# def _crop_random(image):
#     """Randomly crops image and mask"""
#
#     cond_crop = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)   # ***
#     image = tf.cond(cond_crop,
#                     lambda: tf.image.resize_images(tf.random_crop(image, [int(HEIGHT * 0.85), int(WIDTH * 0.85), 4]),
#                                                    size=[HEIGHT, WIDTH]),
#                     lambda: tf.cast(image, tf.float32))
#     return image




# def _iaa_image_mask_op(concat):
#     [concat] = tf.py_func(_iaa_image_fn, [image], [tf.float32])
#     return concat
#
# def _iaa_image_mask_fn(concat):
#
#     aug = iaa.Sequential([
#         iaa.Sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
#     ])
#     return concat



def train_data_fn(data_dir, img_size, num_epochs, batch_size, num_readers=4, num_threads=4):
    # 1
    tfrecords = tf.gfile.Glob(data_dir + str(img_size) + '*')

    # 2
    key_to_features = {
        'image/format': tf.FixedLenFeature((), tf.string, default_value='RAW'),
        'image/shape':tf.FixedLenFeature([3], tf.int64),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'mask/format': tf.FixedLenFeature((), tf.string, default_value='RAW'),
        'mask/shape':tf.FixedLenFeature([3], tf.int64),
        'mask/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    # 3
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', shape=[ORG_SIZE, ORG_SIZE, 3], channels=3),
        'mask': slim.tfexample_decoder.Image('mask/encoded', 'mask/format', shape=[ORG_SIZE, ORG_SIZE, 1], channels=1),
    }

    # 4
    decoder = slim.tfexample_decoder.TFExampleDecoder(key_to_features, items_to_handlers)

    # 5
    dataset = slim.dataset.Dataset(
        data_sources=tfrecords,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=670,
        items_to_descriptions=None,
    )

    # 6
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size,
        shuffle=True,
        num_epochs=num_epochs,
    )

    image, mask = provider.get(['image', 'mask'])

    # 7 processing function
    image = tf.cast(image, tf.float32)      # 0~255
    mask = tf.cast(mask, tf.float32)

    concat = tf.concat([image, mask], axis=-1)
    concat = random_crop_op(concat)
    concat = random_flip_op(concat)
    concat = random_rotate_op(concat)
    concat = elastic_transform_op(concat)
    concat = tf.reshape(concat, [OUT_MASK_SIZE, OUT_MASK_SIZE, 4])  # todo

    image, mask = tf.split(concat, [3, 1], axis=-1)

    # image = togray_op(image)  # todo add it
    # image = random_invert(image)
    # image = random_remap_color(image)
    image = iaa_image_op(image)
    image = tf.reshape(image, [OUT_MASK_SIZE, OUT_MASK_SIZE, 3])    # todo
    image = tf.pad(image, tf.constant([[44, 44,], [44, 44], [0, 0]]), mode='REFLECT')
    # image = distort_color_op(image, color_ordering=1)
    image = tf.clip_by_value(image / 255., 0.0, 1.0)

    mask = tf.clip_by_value(mask / 255., 0.0, 1.0)
    mask = tf.cast(mask>0.5, tf.float32)
    # mask = tf.concat([mask, 1-mask], axis=-1)   # prob of pos & neg # todo add boundary

    return tf.train.batch([image, mask],
                          batch_size=batch_size,
                          num_threads=num_threads,
                          capacity=20*batch_size,
                          shapes=[[OUT_IMG_SIZE, OUT_IMG_SIZE, 3], [OUT_MASK_SIZE, OUT_MASK_SIZE, 1]]
                          )

def load_valid_sample(root_dir, name):
    img_path = os.path.join(root_dir, name, 'images', name + '.png')
    mask_path = glob.glob(os.path.join(root_dir, name, 'masks/') + '*.png')

    # --- image ---
    img_BGR = cv2.imread(img_path)
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    img_YCrCb = cv2.resize(img_YCrCb, (OUT_MASK_SIZE, OUT_MASK_SIZE), interpolation=cv2.INTER_CUBIC)
    # if white convert to black
    # if sum(np.where(img_YCrCb[...,0] > 150)[0].astype(np.bool)) > OUT_MASK_SIZE * OUT_MASK_SIZE * 0.6:  # !
    if np.mean(img_YCrCb[...,0]) > 127.:
        img_YCrCb[..., 0] = 255 - img_YCrCb[..., 0]
    img_RGB = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
    img_RGB = np.pad(img_RGB, ((PAD_SIZE, PAD_SIZE),(PAD_SIZE, PAD_SIZE),(0, 0)), 'reflect')

    # --- mask ---
    mask = np.zeros((OUT_MASK_SIZE, OUT_MASK_SIZE, 1), dtype=np.bool)
    for mask_path_i in mask_path:
        mask_i = cv2.imread(mask_path_i)
        mask_i = cv2.cvtColor(mask_i, cv2.COLOR_BGR2GRAY)
        mask_i = cv2.resize(mask_i, (OUT_MASK_SIZE, OUT_MASK_SIZE))
        mask_i = np.expand_dims(mask_i, axis=-1)
        mask = np.maximum(mask, mask_i)
    mask = mask.astype(np.uint8)

    return img_RGB, mask


def valid_data_fn(data_dir):
    all_ids = next(os.walk(data_dir))[1]
    valid_images = np.zeros((len(all_ids), OUT_IMG_SIZE, OUT_IMG_SIZE, 3), dtype=np.float32)
    valid_masks = np.zeros((len(all_ids), OUT_MASK_SIZE, OUT_MASK_SIZE, 1), dtype=np.float32)

    for i in range(len(all_ids)):
        id_ = all_ids[i]
        image_i, mask_i = load_valid_sample(data_dir, id_)
        valid_images[i] = image_i / 255.
        valid_masks[i] = mask_i / 255.

    return valid_images, valid_masks



if __name__ == '__main__':

    # image, mask = valid_data_fn('./stage1_valid/')
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(image[0])
    # plt.subplot(1,2,2)
    # plt.imshow(np.squeeze(mask[0]), cmap='gray')
    # plt.show()
    # print(mask[1].shape)

    image,mask = train_data_fn('./tfrecords/train/',  240, 10, 10)

    init_op = tf.group(tf.local_variables_initializer(),
                       tf.global_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for _ in range(5):
                print(_)
                img, m = sess.run([image, mask])
                print(img.shape)

        except tf.errors.OutOfRangeError:
            print("catch OutOfRangeError")
        finally:
            coord.request_stop()
            print("finish reading")
        coord.join(threads)
