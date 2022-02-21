# -*- coding: utf-8 -*-
import os
import time
from path import Path
import numpy as np
import tensorflow as tf

Path(__file__).abspath().parent.cd()
import model
# from readimg import read_imgs_1, read_imgs_2
from readimg import read_imgs_file, read_imgs_file_by_group
from processimg import get_min_width_image, to_mnist_image, denoise, random_rotate, random_erose
from dataset import DataSet
from other import chinese_other, right_wrong_other, wrong_other, right_wrong_3_others, right_wrong_4_others, \
    abcdefg_others

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 1000

dataset_train_right_wrong_path = (
    r'./datasets/right-wrong-train-images-idx3-ubyte.gz', r'./datasets/right-wrong-train-labels-idx1-ubyte.gz')
dataset_train_right_wrong_half_right_path = (
    r'./datasets/right-wrong-train-images-idx3-ubyte.gz',
    r'./datasets/right-wrong-half-right-train-labels-idx1-ubyte.gz')
dataset_train_digit_path = (
    r'./datasets/digit-train-images-idx3-ubyte.gz', r'./datasets/digit-train-labels-idx1-ubyte.gz')
dateset_train_abcd = (r'./datasets/abcd-train-images-idx3-ubyte.gz', r'./datasets/abcd-train-labels-idx1-ubyte.gz')
dateset_train_abcdefg = (
r'./datasets/abcdefg-train-images-idx3-ubyte.gz', r'./datasets/abcdefg-train-labels-idx1-ubyte.gz')
mnist_train_path = (r'./datasets/train-images-idx3-ubyte.gz', r'./datasets/train-labels-idx1-ubyte.gz')


def train(label_length, train_type, epoch, groups=None, with_other=False, other_count=1, use_thread=False,
          additional_sample_path=None, additional_fonts=[], additional_fonts_each_group=20):
    max_limit = label_length - 1 if not with_other else label_length - 1 - other_count
    last_merge_other = False
    other_probability = 0
    if max_limit == 0:
        # train for r/w 
        last_merge_other = True
        other_probability = 0.5 
        dataset_path = dataset_train_right_wrong_path
        if not with_other:
            raise ValueError('label_length error')
        # only distinguish between r/w and other
        saver_path = "./model/model-wrong-with-other-{}.ckpt".format(train_type)
        scope = 'model_wrong_with_other'
        other = wrong_other
    elif max_limit == 1:
        dataset_path = dataset_train_right_wrong_path
        if with_other:
            if other_count == 1:
                saver_path = "./model/model-right-wrong-with-other-{}.ckpt".format(train_type)
                scope = 'model_right_wrong_with_other'
                other = right_wrong_other
            else:
                saver_path = "./model/model-right-wrong-with-others-{}.ckpt".format(train_type)
                scope = 'model_right_wrong_with_others'
                other = eval('right_wrong_{}_others'.format(other_count))
        else:
            saver_path = "./model/model-right-wrong-{}.ckpt".format(train_type)
            scope = 'model_right_wrong'
            other = None
    elif max_limit == 2:
        # used
        # train for half right
        dataset_path = dataset_train_right_wrong_half_right_path
        saver_path = "./model/model-right-wrong-with-others-{}.ckpt".format(train_type)
        scope = 'model_right_wrong_with_others'
        other = eval('right_wrong_{}_others'.format(other_count))
    elif max_limit == 3:
        # train for ABCD
        dataset_path = dateset_train_abcd
        saver_path = "./model/model-abcd-{}.ckpt".format(train_type)
        scope = 'model_abcd'
        other = None
    elif max_limit == 6:
        # Train for ABCDEFG
        dataset_path = dateset_train_abcdefg
        saver_path = "./model/model-abcdefg-{}.ckpt".format(train_type)
        scope = 'model_abcdefg'
        other = abcdefg_others
    else:
        # Train for numbers
        dataset_path = dataset_train_digit_path
        if with_other:
            saver_path = "./model/model-number-0-{}-with-other-{}.ckpt".format(max_limit, train_type)
            scope += 'model_number_0_{}_with_other'.format(max_limit)
            other = chinese_other
        else:
            saver_path = "./model/model-number-0-{}-{}.ckpt".format(max_limit, train_type)
            scope = 'model_number_0_{}'.format(max_limit)
            other = None
    # preprocess 
    if train_type == 'base':
        read_middlewares = []
        batch_middlewares = []
    elif train_type == 'min':
        # Shear Affine
        read_middlewares = [get_min_width_image, to_mnist_image]
        batch_middlewares = []
    elif train_type == 'rotate':
        # Rotation
        read_middlewares = []
        if with_other or max_limit == 1:  
            batch_middlewares = [random_rotate, to_mnist_image]
        else:
            batch_middlewares = [random_rotate, get_min_width_image, to_mnist_image]
    elif train_type == 'erose-rotate':
        # Eroed Rotation
        read_middlewares = []
        if with_other or max_limit == 1:
            batch_middlewares = [random_erose, random_rotate, to_mnist_image]
        else:
            batch_middlewares = [random_erose, random_rotate, get_min_width_image, to_mnist_image]
    if label_length <= 10 and not with_other:
        X, Y = read_imgs_file(dataset_path[0], dataset_path[1], label_length, use_thread=use_thread,
                              middlewares=read_middlewares, additional_fonts=additional_fonts,
                              additional_fonts_each_group=additional_fonts_each_group,
                              additional_sample_path=additional_sample_path)
    else:
        if groups is None:
            raise ValueError('groups error')
        X, Y = read_imgs_file_by_group(dataset_path[0], dataset_path[1], label_length, groups=groups, other=other,
                                       other_count=other_count, last_merge_other=last_merge_other,
                                       other_probability=other_probability, use_thread=use_thread,
                                       middlewares=read_middlewares, additional_sample_path=additional_sample_path)

    if Path(dataset_path[0]).name == 'train-images-idx3-ubyte.gz':
        saver_path = Path(saver_path).parent / ('mnist-' + Path(saver_path).name)

    dataset = DataSet(X, Y, use_thread=use_thread, middlewares=batch_middlewares)

    tf.reset_default_graph()
    with tf.variable_scope(scope):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        keep_prob = tf.placeholder(tf.float32)
        y_conv, variables = model.convolutional(x, keep_prob, label_length)

    # update 
    y = tf.placeholder(tf.float32, shape=[None, label_length])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
    prediction = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, saver_path)
        except:
            pass
        try:
            for i in range(epoch):
                batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    now_cross_entropy = cross_entropy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    print('[%s][%s] step %d, accuracy %g, loss %g' % (
                        scope, train_type, i, train_accuracy, now_cross_entropy))
                train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                if i % 500 == 0 and i != 0:
                    saver.save(sess, saver_path)
        finally:
            saver.save(sess, saver_path)


"""
for i in range(1):
    train(10, 'base', 32000, use_thread=True)
    train(10, 'min', 100000, use_thread=True)
    train(10, 'rotate', 100000, use_thread=True)
"""
for i in range(100):
    # r/w
    # train(2, 'rotate', 10000, use_thread=True)
    # r/w and other
    # train(3, 'rotate', 10000, groups=30000, use_thread=True, with_other=True)
    # wrong and other
    # train(2, 'rotate', 10000, groups=30000, use_thread=True, with_other=True)
    # For bold r/2
    # train(6, 'erose-rotate', 20000, groups=20000, use_thread=True, with_other=True, other_count=3, additional_sample_path='/home/zhangxuan/datasets/right-wrong-additional-set')
    # r/w and others (\, /, other)
    # train(6, 'rotate', 20000, groups=20000, use_thread=True, with_other=True, other_count=3, additional_sample_path='/home/zhangxuan/datasets/right-wrong-additional-set')
    # with additional ellipse
    # train(7, 'rotate', 20000, groups=20000, use_thread=True, with_other=True, other_count=4, additional_sample_path='/home/zhangxuan/datasets/right-wrong-additional-set')
    # with support for printing font
    # train(10, 'rotate', 100000, use_thread=True, additional_fonts=['./font/TMSGEO.TTF',], additional_fonts_each_group=20)
    # train(10, 'rotate', 100000, use_thread=True, additional_sample_path='/home/zhangxuan/datasets/digit-additional-set')
    # abcd
    # train(4, 'rotate', 100000, use_thread=True)
    # abcdefg + other
    train(8, 'rotate', 100000, groups=30000, with_other=True, other_count=1, use_thread=True)
