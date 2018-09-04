import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
from scipy.ndimage.interpolation import map_coordinates
from skimage.segmentation import find_boundaries
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d


# todo add to a dict
ORG_SIZE = 240
OUT_IMG_SIZE = 268
OUT_MASK_SIZE = 180
PAD_SIZE = (OUT_IMG_SIZE - OUT_MASK_SIZE) // 2

# - [ ] Grayscale
# - [ ] HSV color shift
# - [ ] GaussianBlur
# - [ ] AdditiveGaussianNoise
# - [ ] random channel shuffle
# - [ ] elastic_transform

def _image_transform(image, coordinates):
    image_trans = image.copy()
    for z in range(image.shape[2]):
        image_trans[:, :, z] = map_coordinates(image[:, :, z], coordinates, order=1, mode='reflect')

    return image_trans

def _mask_transform(mask, coordinates):
    mask_trans = map_coordinates(np.squeeze(mask), coordinates, order=0, mode='reflect')
    return np.expand_dims(mask_trans, axis=-1)

def _elastic_transform_fn(concat, alpha=2000, sigma=30):
    image, mask = np.split(concat, [3], axis=-1)
    h, w = image.shape[:2]

    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode='constant') * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode='constant') * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    yt, xt = y + dy, x + dx

    image_trans = _image_transform(image, (yt, xt))
    mask_trans = _mask_transform(mask, (yt, xt))

    concat = np.concatenate((image_trans, mask_trans), axis=-1)
    return concat

def elastic_transform_op(concat, alpha=2000, sigma=30):
    [concat] = tf.py_func(_elastic_transform_fn, [concat, alpha, sigma], [tf.float32])
    concat = tf.convert_to_tensor(concat, tf.float32)
    return concat


def iaa_image_op(image):
    [image] = tf.py_func(_iaa_image_fn, [image], [tf.float32])
    image = tf.convert_to_tensor(image, tf.float32)
    return image

def _iaa_image_fn(image):

    aug = iaa.Sequential([
        iaa.Sometimes(0.25, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.Sometimes(0.25, iaa.WithColorspace(
            to_colorspace='HSV',
            from_colorspace='RGB',
            children=[
                iaa.WithChannels(0, iaa.Add((-10, 10))),
                iaa.WithChannels(1, iaa.Add((-25, 25))),
                iaa.WithChannels(2, iaa.Multiply((0.8, 1.1)))
            ])),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 3.0))),
        iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=(0, 0.025*255.)))
    ])

    r = aug.augment_image(image.astype(np.uint8))
    c = np.random.shuffle([0, 1, 2])    # shuffle
    return r[..., c].astype(np.float32)


def random_crop_op(image):
    """Randomly crops image and mask"""

    cond_crop = tf.random_uniform([], maxval=6, dtype=tf.int32)
    pred_fn_pairs = {
        tf.equal(cond_crop, tf.constant(0, tf.int32)):
            lambda :tf.random_crop(image, [int(ORG_SIZE * 0.75), int(ORG_SIZE * 0.75), 4]),
        tf.equal(cond_crop, tf.constant(1, tf.int32)):
            lambda :tf.random_crop(image, [int(ORG_SIZE * 0.75), int(ORG_SIZE * 1.00), 4]),
        tf.equal(cond_crop, tf.constant(2, tf.int32)):
            lambda :tf.random_crop(image, [int(ORG_SIZE * 1.00), int(ORG_SIZE * 0.75), 4]),
        tf.equal(cond_crop, tf.constant(3, tf.int32)):
            lambda :tf.random_crop(image, [int(ORG_SIZE * 0.50), int(ORG_SIZE * 0.50), 4]),
        tf.equal(cond_crop, tf.constant(4, tf.int32)):
            lambda :tf.random_crop(image, [int(ORG_SIZE * 0.75), int(ORG_SIZE * 0.50), 4]),
        tf.equal(cond_crop, tf.constant(5, tf.int32)):
            lambda :tf.random_crop(image, [int(ORG_SIZE * 0.50), int(ORG_SIZE * 0.75), 4]),
    }
    image = tf.case(pred_fn_pairs)
    image = tf.image.resize_images(image, size=(OUT_MASK_SIZE, OUT_MASK_SIZE))
    return image

def random_noise_op(image):

    noise = tf.random_normal(image.shape, mean=0.0, stddev=0.0025*1.0, dtype=image.dtype)
    image_noise = image - tf.minimum(noise) + noise

    return image_noise

def random_flip_op(image):
    """Randomly flips image left and right"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image

def random_rotate_op(image):
    """Randomly rotate the image"""

    cond_rotate = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
    div = tf.random_uniform([], minval=1, maxval=3, dtype=tf.int32) # if maxval=5, 45' rot
    radian = tf.constant(np.pi) / tf.cast(div, tf.float32)

    image = tf.cond(cond_rotate,
                    lambda: tf.cast(tf.contrib.image.rotate(image, radian), tf.float32),
                    lambda: tf.cast(image, tf.float32))

    return image

def distort_color_op(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 225.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 225.)

    elif color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 225.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    elif color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 225.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)
