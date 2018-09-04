import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def loss_fn(pred_logistic, true):
    """Loss tensor."""
    # Dice loss based on Jaccard dice score coefficent.

    offset = 1e-5
    corr = tf.reduce_sum(true * pred_logistic, axis=[-3, -2, -1])
    l2_pred = tf.reduce_sum(tf.square(pred_logistic), axis=[-3, -2, -1])
    l2_true = tf.reduce_sum(tf.square(true), axis=[-3, -2, -1])
    dice_coeff = (2. * corr + offset) / (l2_true + l2_pred + offset)

    # Second version: 2-class variant of dice loss
    # corr_inv = tf.reduce_sum((1.- true) * (1.-pred_logistic), axis=[-3, -2, -1])
    # l2_pred_inv = tf.reduce_sum(tf.square(1.-pred_logistic), axis=[-3, -2, -1])
    # l2_true_inv = tf.reduce_sum(tf.square(1.-true), axis=[-3, -2, -1])
    # dice_coeff = ((corr + offset) / (l2_true + l2_pred + offset) +
    #             (corr_inv + offset) / (l2_pred_inv + l2_true_inv + offset))

    loss = tf.subtract(1., tf.reduce_mean(dice_coeff))
    return loss


def prelu(x):
    with tf.variable_scope('prelu'):
        alpha = tf.get_variable('alpha', x.get_shape()[-1],
                                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - abs(x)) * 0.5
        return pos + neg

def unet_small_fn(inputs, is_training=True, scope='unet'):
    with tf.variable_scope(scope, values=[inputs], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            padding='VALID',
                            kernel_size=3,
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(1.0),
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=prelu):
                with slim.arg_scope([slim.max_pool2d], padding='SAME', kernel_size=3):

                    a1 = slim.batch_norm(slim.conv2d(inputs, 24, scope='c1'))# 
                    a2 = slim.batch_norm(slim.conv2d(a1, 24, scope='c2'))# 264
                    p1 = slim.max_pool2d(a2, scope='p1')# 132

                    a3 = slim.batch_norm(slim.conv2d(p1, 48, scope='c3'))
                    a4 = slim.batch_norm(slim.conv2d(a3, 48, scope='c4'))#128
                    p2 = slim.max_pool2d(a4, scope='p2')#64

                    a5 = slim.batch_norm(slim.conv2d(p2, 96, scope='c5'))
                    a6 = slim.batch_norm(slim.conv2d(a5, 96, scope='c6'))#60
                    p3 = slim.max_pool2d(a6, scope='p3')#30

                    a9 = slim.batch_norm(slim.conv2d(p3, 192, scope='c9'))
                    a10 = slim.batch_norm(slim.conv2d(a9, 192, scope='c10'))#26

                    u2 = slim.conv2d_transpose(a10, 96, stride=2, padding='SAME', scope='u2')#52
                    c2 = tf.concat([u2, a6[:,4:-4,4:-4,:]], axis=-1, name='concat2')
                    a13 = slim.batch_norm(slim.conv2d(c2, 96, scope='c13'))
                    a14 = slim.batch_norm(slim.conv2d(a13, 96, scope='c14'))

                    u3 = slim.conv2d_transpose(a14, 48, stride=2, padding='SAME', scope='u3')
                    c3 = tf.concat([u3, a4[:,16:-16,16:-16,:]], axis=-1, name='concat3')
                    a15 = slim.batch_norm(slim.conv2d(c3, 48, scope='c15'))
                    a16 = slim.batch_norm(slim.conv2d(a15, 48, scope='c16'))

                    u4 = slim.conv2d_transpose(a16, 24, stride=2, padding='SAME', scope='u4')
                    c4 = tf.concat([u4, a2[:,40:-40,40:-40,:]], axis=-1, name='concat4')
                    a17 = slim.batch_norm(slim.conv2d(c4, 24, scope='c17'))
                    a18 = slim.conv2d(a17, 1, scope='c18')

                    pred = a18
                    pred_sig = tf.nn.sigmoid(a18)
                    # pred_logistic = slim.batch_norm(pred, activation_fn=tf.nn.sigmoid)
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                    return pred, pred_sig, end_points


# def unet_fn(inputs, is_training=True, scope='unet'):
#     with tf.variable_scope(scope, values=[inputs], reuse=tf.AUTO_REUSE) as sc:
#         end_points_collection = sc.name + '_end_points'
#         with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
#                             padding='SAME',
#                             kernel_size=3,
#                             activation_fn=None,
#                             weights_initializer=tf.contrib.layers.xavier_initializer(),
#                             weights_regularizer=slim.l2_regularizer(1.0),
#                             outputs_collections=[end_points_collection]):
#             with slim.arg_scope([slim.batch_norm],
#                                 is_training=is_training,
#                                 activation_fn=prelu):
#                 with slim.arg_scope([slim.max_pool2d], padding='SAME', kernel_size=3):
#
#                     a1 = slim.batch_norm(slim.conv2d(inputs, 16, scope='c1'))
#                     a2 = slim.batch_norm(slim.conv2d(a1, 16, scope='c2')) # 128, 128, 16
#                     p1 = slim.max_pool2d(a2, scope='p1')
#
#                     a3 = slim.batch_norm(slim.conv2d(p1, 32, scope='c3'))
#                     a4 = slim.batch_norm(slim.conv2d(a3, 32, scope='c4')) # 64, 64, 32
#                     p2 = slim.max_pool2d(a4, scope='p2')
#
#                     a5 = slim.batch_norm(slim.conv2d(p2, 64, scope='c5'))
#                     a6 = slim.batch_norm(slim.conv2d(a5, 64, scope='c6')) # 32, 32, 64
#                     p3 = slim.max_pool2d(a6, scope='p3')
#
#                     a7 = slim.batch_norm(slim.conv2d(p3, 128, scope='c7'))
#                     a8 = slim.batch_norm(slim.conv2d(a7, 128, scope='c8')) # 16, 16, 128
#                     p4 = slim.max_pool2d(a8, scope='p4')
#
#                     a9 = slim.batch_norm(slim.conv2d(p4, 256, scope='c9'))
#                     a10 = slim.batch_norm(slim.conv2d(a9, 256, scope='c10'))
#
#                     u1 = slim.conv2d_transpose(a10, 128, stride=2, scope='u1') # 16, 16, 128
#                     c1 = tf.concat([u1, a8], axis=-1, name='concat1')
#                     a11 = slim.batch_norm(slim.conv2d(c1, 128, scope='c11'))
#                     a12 = slim.batch_norm(slim.conv2d(a11, 128, scope='c12'))
#
#                     u2 = slim.conv2d_transpose(a12, 64, stride=2, scope='u2') # 32, 32, 64
#                     c2 = tf.concat([u2, a6], axis=-1, name='concat2')
#                     a13 = slim.batch_norm(slim.conv2d(c2, 64, scope='c13'))
#                     a14 = slim.batch_norm(slim.conv2d(a13, 64, scope='c14'))
#
#                     u3 = slim.conv2d_transpose(a14, 32, stride=2, scope='u3') # 64, 64, 32
#                     c3 = tf.concat([u3, a4], axis=-1, name='concat3')
#                     a15 = slim.batch_norm(slim.conv2d(c3, 32, scope='c15'))
#                     a16 = slim.batch_norm(slim.conv2d(a15, 32, scope='c16'))
#
#                     u4 = slim.conv2d_transpose(a16, 16, stride=2, scope='u4') # 128, 128, 16
#                     c4 = tf.concat([u4, a2], axis=-1, name='concat4')
#                     a17 = slim.batch_norm(slim.conv2d(c4, 16, scope='c17'))
#                     a18 = slim.conv2d(a17, 1, scope='c18')
#
#                     pred = a18
#                     pred_sig = tf.nn.sigmoid(a18)
#                     # pred_logistic = slim.batch_norm(pred, activation_fn=tf.nn.sigmoid)
#                     end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#
#                     return pred, pred_sig, end_points



if __name__ == "__main__":
    fake_image = np.random.rand(1, 128, 128, 1)

    input = tf.placeholder(tf.float32, [None, 128, 128, 1])
    pred, pred_log, end_points = unet(input, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./logs/', tf.get_default_graph())
        out, feature_maps = sess.run([pred, end_points], feed_dict={input:fake_image})
        print(out.shape)
        print(feature_maps.keys())
