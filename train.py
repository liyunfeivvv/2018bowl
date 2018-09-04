import os
import csv
import time
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from utils.vis_fn import vis_fn
from utils.metric import get_score
from net.unet import unet_small_fn, loss_fn
from input.data_factory import train_data_fn, valid_data_fn


flags = tf.app.flags
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('wd', 1e-5, 'weight decay')
flags.DEFINE_string('lr_mode', 'piecewise_constant',
                    'constant / exponential_decay / natural_exp_decay / piecewise_constant / polynomial_decay')
flags.DEFINE_string('opt_mode', 'mom',
                    'sgd / mom / rms / adam /')
flags.DEFINE_string('start_with', 'restore',
                    'sketch / restore')
flags.DEFINE_integer('bs', 32, 'batch size')
flags.DEFINE_integer('ep', 500, 'number epochs')
flags.DEFINE_integer('ep_size', 670, 'the num of train step of one epoch')
flags.DEFINE_integer('train_dataset_size', 240, '')

flags.DEFINE_string('train_tfrecord_path', './input/tfrecords/train/', '')
flags.DEFINE_string('valid_path', './input/stage1_valid/', '')
flags.DEFINE_string('checkpoint_dir', './checkpoint/08-29-19-42-13/', '')
flags.DEFINE_string('log_dir', './logs/', '')

flags.DEFINE_float('gpu_memory_fraction', 0.9, '')
flags.DEFINE_boolean('allow_soft_placement', True, '')
flags.DEFINE_boolean('allow_growth', True, '')

FLAGS = flags.FLAGS


def main(_):
    symbol = time.strftime("%m-%d-%H-%M-%S", time.localtime())

    # with tf.Graph().as_default():
    # define train data
    with tf.device('/cpu:0'):
        with tf.name_scope('get_train_data'):
            train_imgs, train_masks = train_data_fn(FLAGS.train_tfrecord_path, FLAGS.train_dataset_size, FLAGS.ep, FLAGS.bs)

    # define train net
    pred, pred_sig, end_points = unet_small_fn(train_imgs)
    tf.summary.image('inputs', train_imgs, max_outputs=4)
    tf.summary.image('labels', train_masks, max_outputs=4)
    tf.summary.image('pred_sig', pred_sig, max_outputs=4)

    # define valid data
    valid_imgs_op = tf.placeholder(tf.float32, [None, 268, 268, 3])# todo

    # define valid net
    _, valid_pred_op, __ = unet_small_fn(valid_imgs_op, is_training=False)

    # define loss function
    dice_loss = loss_fn(pred_sig, train_masks)
    l2_loss = FLAGS.wd * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = dice_loss + l2_loss
    tf.summary.scalar('loss', loss)

    # define global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # define learning rate
    if FLAGS.lr_mode == 'constant':
        learning_rate = FLAGS.lr
    elif FLAGS.lr_mode == 'exponential_decay':
        decay_steps = FLAGS.ep_size * 20
        decay_rate = 0.9
        staircase = True
        learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, decay_steps, decay_rate, staircase)
    elif FLAGS.lr_mode == 'piecewise_constant':
        boundaries = [FLAGS.ep_size * i for i in [100, 200, 350]]
        values = [FLAGS.lr * i for i in [1.0, 0.5, 0.1, 0.05]]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    elif FLAGS.lr_mode == 'polynomial_decay':
        decay_steps = FLAGS.ep_size * 20
        end_learning_rate = 0.01 * FLAGS.lr
        power = 1.0
        cycle = True
        learning_rate = tf.train.polynomial_decay(FLAGS.lr, global_step, decay_steps, end_learning_rate, power, cycle)
    else:
        raise ValueError('Unsupported Learning Rate Mode:{}'.format(FLAGS.lr_mode))
    tf.summary.scalar("learning_rate", learning_rate)

    # define optimizer
    if FLAGS.opt_mode == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.opt_mode == 'mom':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif FLAGS.opt_mode == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9)
    elif FLAGS.opt_mode == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError('Unsuported Optimizer{}'.format(FLAGS.opt_mode))

    # define trained variable
    train_vars = tf.trainable_variables()

    # define train op
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        grads_and_vars = optimizer.compute_gradients(loss, train_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    # define merged summary
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + symbol + '/tensorboard/', tf.get_default_graph())

    # define session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction, allow_growth=FLAGS.allow_growth)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=FLAGS.allow_soft_placement))

    # define loaded variables
    load_vars = tf.global_variables()
    loader = tf.train.Saver(load_vars)

    # define saved variables
    save_vars = tf.global_variables()
    saver = tf.train.Saver(save_vars)

    # define init op
    if FLAGS.start_with == 'sketch':
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        print("From sketch.")
    elif FLAGS.start_with == 'restore':
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        checkpoint_model = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        checkpoint_load = checkpoint_model.model_checkpoint_path
        loader.restore(sess, checkpoint_load)
        print("Restore from :", checkpoint_load)
    else:
        raise ValueError("Not a reasonable option")

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            print('====== Start Training ======')
            
            valid_imgs, valid_masks = valid_data_fn(FLAGS.valid_path)
            valid_feed_dict = {valid_imgs_op: valid_imgs}
            
            train_losses, train_scores, valid_scores = [], [], []
            max_score, best_ep, count = 0, 0, 0

            for ep in range(FLAGS.ep):
                # train
                for s in range(FLAGS.ep_size // FLAGS.bs):
                    l, _ = sess.run([loss, train_op])
                train_mask_arr, train_pred_arr, l, m, gs = sess.run([train_masks, pred_sig, loss, merged, global_step])
                train_score = np.mean(get_score(train_mask_arr, train_pred_arr))
                train_losses.append(l)
                train_scores.append(train_score.item())

                # valid
                valid_pred = sess.run(valid_pred_op, valid_feed_dict)
                valid_score = np.mean(get_score(valid_masks, valid_pred))
                valid_scores.append(valid_score.item())

                # summary
                train_writer.add_summary(m, global_step=gs)
                print("Epoch:{:4d}, train loss:{:.4f}, train score:{:.4f}, valid score:{:.4f}"
                      .format(ep, l, train_score, valid_score))

                # show or save
                vis_fn(valid_imgs, valid_masks, valid_pred, True, FLAGS.log_dir + symbol + '/' , ep, 5, False)

                # save model
                if (ep + 1) % 20 == 0:
                    saver.save(sess, FLAGS.checkpoint_dir + symbol + '/' + 'unet.ckpt', global_step=ep)
                if max_score < valid_score:
                    max_score = valid_score
                    best_ep = ep
                    count = 0
                    saver.save(sess, FLAGS.checkpoint_dir + symbol + '/' + 'best_unet.ckpt', global_step=ep)
                if count > 50 and ep > 270:   
                    saver.save(sess, FLAGS.checkpoint_dir + symbol + '/' + 'last_unet.ckpt', global_step=ep)
                    coord.request_stop()
                count += 1

            print('====== End ======')

        except tf.errors.OutOfRangeError:
            print("catch OutOfRangeError")

        finally:
            print("The best epoch:{}".format(best_ep))

            if not os.path.exists(FLAGS.log_dir + symbol + '/' + 'metric'):
                os.makedirs(FLAGS.log_dir + symbol + '/' + 'metric')

            with open(FLAGS.log_dir + symbol + '/' + 'metric/train_log.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_score', 'valid_score'])
                for a, b, c, d in zip(range(1, len(train_losses)+1), train_losses, train_scores, valid_scores):
                    writer.writerow([a, b, c, d])

            coord.request_stop()
            print('finish reading')

        coord.join(threads)
        train_writer.close()


if __name__ == '__main__':
    tf.app.run()
