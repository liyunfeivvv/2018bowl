import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv
import tqdm

from skimage.morphology import closing, opening, disk, label, binary_dilation
#from skimage.io import imread, imsave
#from skimage.transform import resize
import cv2
from net.unet import unet_small_fn

flags = tf.app.flags

flags.DEFINE_string('test_dir', './input/stage2_test_final/', '')
flags.DEFINE_string('checkpoint_dir', './checkpoint/08-29-19-42-13/', '')
flags.DEFINE_string('retore_mode', 'best', 'best / last')

flags.DEFINE_boolean('save_img', False, '')
flags.DEFINE_string('save_dir', './input/', '')

flags.DEFINE_integer('img_size', 180, '')
flags.DEFINE_integer('input_size', 268, '')

FLAGS = flags.FLAGS

# *************************************************** #
def clean_img(x):
    return opening(closing(x, disk(1)), disk(3))

def test_data_fn(root_dir, name, size, pad):
    img_path = '{}{}/images/{}.png'.format(root_dir, name, name)
    
    img_BGR = cv2.imread(img_path)
    assert len(img_BGR.shape) == 3
    assert img_BGR.shape[2] == 3
    
    org_h, org_w = img_BGR.shape[:2]
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    img_YCrCb = cv2.resize(img_YCrCb, (size, size), interpolation=cv2.INTER_CUBIC)

    if np.mean(img_YCrCb[...,0])>127.:  # !
        img_YCrCb[..., 0] = 255 - img_YCrCb[..., 0]
    
    img_RGB = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)
    img_RGB = np.pad(img_RGB, ((pad, pad),(pad, pad),(0, 0)), 'reflect')

    return img_RGB, org_h, org_w
    

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5, dilation=True):
    lab_img = label(x > cutoff) # split of components goes here
    if dilation:
        for i in range(1, lab_img.max() + 1):
            lab_img = np.maximum(lab_img, binary_dilation(lab_img==i)*i)
    for i in range(1, lab_img.max() + 1):
        img = lab_img == i
        yield rle_encoding(img)
# *************************************************** #


def main(_):
    save_imgs_dir = FLAGS.save_dir + 'test_result_images/'
    mkdir(save_imgs_dir)

    # load net
    imgs_op = tf.placeholder(tf.float32, [None, FLAGS.input_size, FLAGS.input_size, 3])
    _, pred_op, __ = unet_small_fn(imgs_op, is_training=False)


    loader = tf.train.Saver()
    sess = tf.Session()
    
    # load the last
    checkpoint_model = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    checkpoint_paths = checkpoint_model.all_model_checkpoint_paths
    if FLAGS.retore_mode == 'best':
        for path in checkpoint_paths[::-1]:
            if 'best' in path:
                checkpoint_load = path
                break
    else:
        for path in checkpoint_paths[::-1]:
            if 'last' in path:
                checkpoint_load = path
                break
    loader.restore(sess, checkpoint_load)
    print("Restore from :", checkpoint_load)
    
    pad = (FLAGS.input_size - FLAGS.img_size) // 2
    new_all_ids = []
    all_rles = []
    all_ids = tf.gfile.ListDirectory(FLAGS.test_dir)
    for id_ in tqdm.tqdm(all_ids):
        img, org_h, org_w = test_data_fn(FLAGS.test_dir, id_, FLAGS.img_size, pad)
        img = img[np.newaxis, ...]
        img = img.astype(np.float32) / 255.

        pred = sess.run(pred_op, feed_dict={imgs_op:img})
        
        pred = np.squeeze(pred)
        pred = cv2.resize(pred, (org_w, org_h), interpolation=cv2.INTER_CUBIC)
        pred = clean_img(pred)

        if FLAGS.save_img:
            save_path = save_imgs_dir + id_ + 'pred.jpg'
            cv2.imwrite(save_path, pred)

        rle = list(prob_to_rles(pred))
        all_rles.extend(rle)
        new_all_ids.extend([id_] * len(rle))

    # with open('submission2.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['ImageId', 'EncodedPixels'])
    #     for a, b in all_rles:
    #         writer.writerow([a, b])

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_all_ids
    sub['EncodedPixels'] = pd.Series(all_rles).apply(lambda x: ' '.join(str(y) for y in x))

    fname = FLAGS.retore_mode + FLAGS.checkpoint_dir.split('/')[-2]
    print("Submission file: " + fname)
    sub.to_csv(fname, index=False)

if __name__ == '__main__':
    tf.app.run()







# # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# def rle_encode(img, min_threshold=1e-3, max_threshold=None):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     if np.max(img) < min_threshold:
#         return '' ## no need to encode if it's all zeros
#     if max_threshold and np.mean(img) > max_threshold:
#         return '' ## ignore overfilled mask
#     pixels = img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)
#
# def rle_encode(mask):
#     pixels = mask.T.flatten()
#     # We need to allow for cases where there is a '1' at either end of the sequence.
#     # We do this by padding with a zero at each end when needed.
#     use_padding = False
#     if pixels[0] or pixels[-1]:
#         use_padding = True
#         pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
#         pixel_padded[1:-1] = pixels
#         pixels = pixel_padded
#     rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     if use_padding:
#         rle = rle - 1
#     rle[1::2] = rle[1::2] - rle[:-1:2]
#     return rle
#
#
# def rle_encoding(x):
#     '''
#     x: numpy array of shape (height, width), 1 - mask, 0 - background
#     Returns run length as list
#     '''
#     dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b > prev + 1): run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths
#
#
# def prob_to_rles(x, cut_off=0.5):
#     lab_img = label(x > cut_off)
#     if lab_img.max() < 1:
#         lab_img[0, 0] = 1  # ensure at least one prediction per image
#     for i in range(1, lab_img.max() + 1):
#         yield rle_encoding(lab_img == i)



# ref.: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if type(mask_rle) == str:
        s = mask_rle.split()
    else:
        s = mask_rle
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

