# -*- coding: utf-8 -*-
import os
from path import Path
import numpy as np
import tensorflow as tf
import cv2

Path(__file__).parent.abspath().cd()
from readimg import read_imgs_file, read_imgs_file_by_group
from processimg import get_min_width_image, to_mnist_image, denoise, random_rotate
import model
from other import chinese_other, right_wrong_other

BATCH_SIZE = 500

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

dataset_train_digit_path = (r'./datasets/digit-train-images-idx3-ubyte.gz', r'./datasets/digit-train-labels-idx1-ubyte.gz')

dataset_test_right_wrong_path = (r'./datasets/right-wrong-test-images-idx3-ubyte.gz', r'./datasets/right-wrong-test-labels-idx1-ubyte.gz')
dataset_test_digit_path = (r'./datasets/digit-test-images-idx3-ubyte.gz', r'./datasets/digit-test-labels-idx1-ubyte.gz')
mnist_test_path = (r'./datasets/t10k-images-idx3-ubyte.gz', r'./datasets/t10k-labels-idx1-ubyte.gz')
#dataset_test_digit_path = mnist_test_path

def test(label_length, train_type, groups=None, with_other=False, is_mnist_dataset=False, use_thread=False, save_error=True):
    max_limit = label_length-1 if not with_other else label_length-2
    if max_limit == 1:
        dataset_path = dataset_test_right_wrong_path
        if with_other:
            saver_path = "./model/model-right-wrong-with-other-{}.ckpt".format(train_type)
            scope = 'model_right_wrong_with_other'
            other = right_wrong_other
        else:
            saver_path = "./model/model-right-wrong-{}.ckpt".format(train_type)
            scope = 'model_right_wrong'
            other = None
    else:
        dataset_path = dataset_test_digit_path
        if with_other:
            saver_path = "./model/model-number-0-{}-with-other-{}.ckpt".format(max_limit, train_type)
            scope = 'model_number_0_{}_with_other'.format(max_limit)
            other = chinese_other
        else:
            saver_path = "./model/model-number-0-{}-{}.ckpt".format(max_limit, train_type)
            scope = 'model_number_0_{}'.format(max_limit)
            other = None
    if train_type == 'base':
        read_middlewares = []
    elif train_type == 'min':
        read_middlewares = [get_min_width_image, to_mnist_image]
    elif train_type == 'rotate':
        read_middlewares = [random_rotate, get_min_width_image, to_mnist_image]
    if 2 <= max_limit < 9:
        raise ValueError('label_length error')
    if label_length <= 10 and not with_other:
        X, Y = read_imgs_file(dataset_path[0], dataset_path[1], label_length, use_thread=use_thread, middlewares=read_middlewares)
    else:
        if groups is None:
            raise ValueError('groups error')
        X, Y = read_imgs_file_by_group(dataset_path[0], dataset_path[1], label_length, groups=groups, other=other, use_thread=use_thread, middlewares=read_middlewares)
    print('use test set: ', dataset_path[0])
    if is_mnist_dataset:
        saver_path = Path(saver_path).parent / ('mnist-' + Path(saver_path).name)
    float_X = np.multiply(X.astype(np.float32), 1.0 / 255.0)
    tf.reset_default_graph()
    with tf.variable_scope(scope):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        keep_prob = tf.placeholder(tf.float32)
        y_conv, _ = model.convolutional(x, keep_prob, label_length)
        prediction = tf.argmax(y_conv, 1)

    y = tf.placeholder(tf.float32, shape=[None, label_length])
    correct_prediction = tf.equal(prediction, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, saver_path)
        #float_X = float_X[:1000]
        test_length = len(float_X)
        arg_pred_wrong = {}
        for i in range(0, test_length, BATCH_SIZE):
            batch_X = float_X[i:i+BATCH_SIZE]
            batch_Y = Y[i:i+BATCH_SIZE]
            batch_pred = sess.run(prediction, feed_dict={x: batch_X.reshape(-1, 784), keep_prob: 1.0})
            batch_pred_is_right = sess.run(tf.equal(batch_pred, tf.argmax(batch_Y, 1)))
            batch_arg_pred_wrong = np.argwhere(batch_pred_is_right==False).flatten()
            arg_pred_wrong.update({(arg+i, batch_pred[arg]) for arg in batch_arg_pred_wrong})

        """
        pred = sess.run(prediction, feed_dict={x: float_X[:1000].reshape(-1, 784), keep_prob: 1.0})
        preds_is_right = sess.run(tf.equal(pred, tf.argmax(Y[:1000], 1)))
        arg_preds = np.argwhere(preds_is_right==False).flatten()
        test_accuracy = 1 - len(arg_preds) / 1000
        """
        test_accuracy = 1 - len(arg_pred_wrong) / test_length
        d = Path('./test_error/{}-{}/'.format(scope, train_type))
        d.rmdir_p()
        d.makedirs_p()
        for wrong_id, wrong_pred in arg_pred_wrong.items():
            cv2.imwrite(d/'{}-{}.png'.format(wrong_id, wrong_pred), X[wrong_id])
        print('[%s][%s] accuracy %g' % (scope, train_type, test_accuracy))


def test_by_dir(test_dir_path, error_dir_path, model_path, label_length, scope, error_dir_name, half_right=-1):
    sess = tf.Session()
    with tf.variable_scope(scope, reuse=False):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        keep_prob = tf.placeholder(tf.float32)
        y_conv, _ = model.convolutional(x, keep_prob, label_length, False)
        prediction = tf.argmax(y_conv, 1)
        
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
    saver.restore(sess, model_path)
    for _dir in Path(test_dir_path).dirs():
        label = int(_dir.name[0])
        if 'error' in _dir.name:
            continue
        dir_name = _dir.name
        error_dir = Path(error_dir_path) / '{}-{}-error'.format(dir_name, error_dir_name)
        error_dir.makedirs_p()
        for img_path in _dir.files():
            img = cv2.imread(img_path, 0)
            print(img_path)
            img_original = img.copy()
            k_size = max(1, max(*img.shape) // 23)
            img = cv2.dilate(img, np.ones((k_size, k_size), dtype='uint8'), iterations=1)
            img = denoise(img, max_class=False)
            try:
                img = to_mnist_image(img)
            except:
                continue
            X = np.multiply(img.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
            pred = sess.run(prediction, feed_dict={x: X, keep_prob: 1.0}).tolist()[0]

            if (label == 0 and pred not in (0, half_right)) or (label != 0 and pred in (0, half_right)):
                img_name = img_path.name.stripext()
                print(img_path.name, 'pred: ', pred, 'actual: ', label)
                ext = img_path.ext
                cv2.imwrite(error_dir / "{}_{}{}".format(img_name, pred, ext), img)
                cv2.imwrite(error_dir / "{}_original_{}{}".format(img_name, pred, ext), img_original)

#test(10, 'base', use_thread=True, is_mnist_dataset=False)
#test(10, 'min', use_thread=True, is_mnist_dataset=False)
#test(10, 'rotate', use_thread=True, is_mnist_dataset=False)
#test(3, 'base', with_other=True, groups=2000)

#test_by_dir(r'D:\datasets\right-wrong-test-set2', './model/model-wrong-with-other-rotate.ckpt', 2, 'model_wrong_with_other', 'model1')
#test_by_dir(r'D:\datasets\right-wrong-test-set2', './model/model-right-wrong-with-other-rotate.ckpt', 3, 'model_right_wrong_with_other', 'model2')
#test_by_dir(r'D:\datasets\right-wrong-test-set4', './model/model-right-wrong-with-others-rotate.ckpt', 6, 'model_right_wrong_with_others', 'model3', half_right=2)
#test_by_dir(r'D:\datasets\huangpin-set', r'D:\datasets\huangpin-after-model0118', './model/backup20190118/model-right-wrong-with-others-rotate.ckpt', 6, 'model_right_wrong_with_others', 'model5', half_right=2)
#test_by_dir(r'D:\datasets\huangpin-set', r'D:\datasets\huangpin-after-model0429', './model/backup20190429/model-right-wrong-with-others-rotate.ckpt', 6, 'model_right_wrong_with_others', 'model5', half_right=2)
#test_by_dir(r'D:\datasets\huangpin-set', r'D:\datasets\huangpin-after-model0505', './model/backup20190505/model-right-wrong-with-others-rotate.ckpt', 6, 'model_right_wrong_with_others', 'model5', half_right=2)
#test_by_dir(r'D:\datasets\huangpin-set', r'D:\datasets\huangpin-after-model0505-1', './model/backup20190505-1/model-right-wrong-with-others-rotate.ckpt', 6, 'model_right_wrong_with_others', 'model5', half_right=2)
test_by_dir(r'D:\datasets\huangpin-set', r'D:\datasets\huangpin-after-model0506', './model/backup20190506/model-right-wrong-with-others-rotate.ckpt', 6, 'model_right_wrong_with_others', 'model5', half_right=2)