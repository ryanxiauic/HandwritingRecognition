# -*- coding: utf-8 -*-
import sys
import os
import base64
from pprint import pprint
import tensorflow as tf
from path import Path
from eve import Eve
from flask import request, jsonify
import cv2
import numpy as np
from datetime import datetime
from functools import reduce
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ALL_COMPLETED
sys.path.append('cnn')
from processimg import process_img

sys.path.append('cnn')
import model
from processimg import process_img, is_blank

SETTINGS = {
      'DOMAIN': {'people': {}},
}

DEBUG = 'HWD_DEBUG' in os.environ
#DEBUG = True

parent = Path(__file__).abspath().parent
parent.cd()
Path('debug').mkdir_p()

pool = ProcessPoolExecutor(max_workers=16)

app = Eve(settings=SETTINGS)
sess = tf.Session()

LABEL_LENGTH = 10

if LABEL_LENGTH == 52:
    saver_path = "./cnn/model/model-number-0-50-with-other.ckpt"
    scope = 'model_number_0_50_with_other'
elif LABEL_LENGTH == 51:
    saver_path = "./cnn/model/model-number-0-50.ckpt"
    scope = 'model_number_0_50'
elif LABEL_LENGTH == 10:
    saver_path = "./cnn/model/model-number-0-9-rotate.ckpt"
    scope = 'model_number_0_9'


with tf.variable_scope(scope):
    x_ = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_ = tf.placeholder(tf.float32)
    y_conv_number, variables_number = model.convolutional(x_, keep_prob_, LABEL_LENGTH, False)
    prediction_number = tf.argmax(y_conv_number, 1)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
saver.restore(sess, saver_path)


with tf.variable_scope('model_number_print_0_9', reuse=False):
    x_print = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_print = tf.placeholder(tf.float32)
    y_conv_print, variables_print = model.convolutional(x_print, keep_prob_print, LABEL_LENGTH, False)
    prediction_print = tf.argmax(y_conv_print, 1)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_number_print_0_9'))
saver.restore(sess, "./cnn/model/model-number-print-0-9-rotate.ckpt")


with tf.variable_scope('model_right_wrong', reuse=False):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y_conv_right_wrong, _ = model.convolutional(x, keep_prob, 2, False)
    prediction_right_wrong = tf.argmax(y_conv_right_wrong, 1)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_right_wrong'))
saver.restore(sess, "./cnn/model/model-right-wrong-rotate.ckpt")


with tf.variable_scope('model_right_wrong_with_others', reuse=False):
    x_right_wrong_others = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_right_wrong_others = tf.placeholder(tf.float32)
    y_conv_right_wrong_with_others, _ = model.convolutional(x_right_wrong_others, keep_prob_right_wrong_others, 7, False)
    prediction_right_wrong_with_others = tf.argmax(y_conv_right_wrong_with_others, 1)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_right_wrong_with_others'))
saver.restore(sess, "./cnn/model/model-right-wrong-with-others-rotate.ckpt")


with tf.variable_scope('model_abcd', reuse=False):
    x_abcd = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_abcd = tf.placeholder(tf.float32)
    y_conv_abcd, _ = model.convolutional(x_abcd, keep_prob_abcd, 4, False)
    prediction_abcd = tf.argmax(y_conv_abcd, 1)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_abcd'))
saver.restore(sess, './cnn/model/model-abcd-rotate.ckpt')

with tf.variable_scope('model_abcdefg', reuse=False):
    x_abcdefg = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_abcdefg = tf.placeholder(tf.float32)
    y_conv_abcdefg, _ = model.convolutional(x_abcdefg, keep_prob_abcdefg, 8, False)
    prediction_abcdefg = tf.argmax(y_conv_abcdefg, 1)

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model_abcdefg'))
saver.restore(sess, './cnn/model/model-abcdefg-rotate.ckpt')

def save_debug_info(pred, img, img_original):
    now_date = datetime.now().strftime("%Y-%m-%d")
    now_date_debug_dir = Path('debug') / now_date
    now_date_debug_dir.mkdir_p()
    now_time = datetime.now().strftime("%H-%M-%S-%f")
    if img is not None:
        cv2.imwrite('{}/{}-{}.png'.format(now_date_debug_dir, now_time, pred), img)
    cv2.imwrite('{}/{}-{}-original.png'.format(now_date_debug_dir, now_time, pred), img_original)


def img_base64_to_array(img_base64, border=True, line=None, is_black=True, img_type='digit', clear_noise=True):
    try:
        img_str = base64.decodestring(str.encode(img_base64))
        img_original = cv2.imdecode(np.fromstring(img_str, np.uint8), cv2.IMREAD_GRAYSCALE)
        return process_img(img_original, border, line, is_black, img_type, clear_noise), img_original
    except:
        save_debug_info('error', None, img_original)
        return np.zeros((28, 28), dtype='uint8'), img_original


@app.route('/readAnswerPoint', methods=['POST'])
def read_answer_point():
    """
    Recognize a single number
    """
    image = request.json.get('image')
    line = request.json.get('line')
    # border不传默认为True
    border = request.json.get('border') if request.json.get('border') is not None else True
    # is_black不传默认为False
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else False
    try:
        img, img_original= img_base64_to_array(image, border=border, line=line, is_black=is_black, img_type='digit')
    except:
        return jsonify({'success': False, 'point': None, 'confidence':0.0})
    if is_blank(img):
        return jsonify({'success': True, 'point': [0,], 'confidence':1.0})
    X = np.multiply(img.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    pred = int(sess.run(prediction_number, feed_dict={x_: X, keep_prob_: 1.0})[0])
    real = sess.run(y_conv_number, feed_dict={x_: X, keep_prob_: 1.0}).tolist()[0]
    confidence = np.power(max(real), 2) / reduce(lambda x, y: x+y, [i*i for i in real if i>0])
    if pred == 1:
        confidence = np.power(confidence, 1/2)
    elif pred in (2, 5):
        confidence = np.power(confidence, 1/3)
    elif pred == 9:
        confidence = np.power(confidence, 1/20)
    if DEBUG:
        save_debug_info(pred, img, img_original)
    return jsonify({'success': True, 'point': [pred,], 'confidence': round(confidence,3)})


@app.route('/readAnswerRightWrong', methods=['POST'])
def read_answer_right_wrong():
    """
    Recognize r/w
    """
    image = request.json.get('image')
    line = request.json.get('line')
    # border不传默认为True
    border = request.json.get('border') if request.json.get('border') is not None else True
    # is_black不传默认为False
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else False
    try:
        img, img_original = img_base64_to_array(image, border=border, line=line, is_black=is_black, img_type='right_wrong', clear_noise=False)
    except:
        return jsonify({'success': False, 'isRight': False})
    if is_blank(img):
        return jsonify({'success': True, 'isRight': True})
    X = np.multiply(img.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    pred = int(sess.run(prediction_right_wrong, feed_dict={x: X, keep_prob: 1.0})[0])
    if DEBUG:
        save_debug_info(pred, img, img_original)
    return jsonify({'success': True, 'isRight': pred==1})


@app.route('/readAnswerPoints', methods=['POST'])
def read_answer_points():
    """
    recognize multiple number
    """
    images = request.json.get('images')
    line = request.json.get('line')
    border = request.json.get('border') if request.json.get('border') is not None else False
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else True
    imgs = []
    all_imgs = []
    blanks = []
    try:
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'digit', True) for i in range(len(images))]
    except:
        global pool
        pool = ProcessPoolExecutor(max_workers=16)
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'digit', True) for i in range(len(images))]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    for i, future in enumerate(futures):
        img, img_original = future.result()
        if is_blank(img):
            blanks.append(i)
        else:
            imgs.append(img)
        all_imgs.append((img, img_original))
    imgs = np.array(imgs, dtype=np.uint8)
    X = np.multiply(imgs.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    preds = sess.run(prediction_number, feed_dict={x_: X, keep_prob_: 1.0}).tolist()
    point_list = [{'number': pred} for pred in preds]
    for i in blanks:
        point_list.insert(i, {'number': 0, 'isBlank': True})
    if DEBUG:
        for i in range(len(point_list)):
            point = point_list[i]['number']
            if 'isBlank' in point_list[i].keys():
                point = str(point) + '_blank'
            save_debug_info('id{}_{}'.format(i, point), all_imgs[i][0], all_imgs[i][1])

    return jsonify({'success': True, 'point': point_list})


@app.route('/readAnswerPointsWithPrint', methods=['POST'])
def read_answer_points_with_print():
    """
    recognize number in both printing type or hand writting
    """
    images = request.json.get('images')
    line = request.json.get('line')
    # border不传默认为False
    border = request.json.get('border') if request.json.get('border') is not None else False
    # is_black不传默认为True
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else True
    imgs = []
    all_imgs = []
    blanks = []
    try:
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'digit', True) for i in range(len(images))]
    except:
        global pool
        pool = ProcessPoolExecutor(max_workers=16)
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'digit', True) for i in range(len(images))]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    for i, future in enumerate(futures):
        img, img_original = future.result()
        if is_blank(img):
            blanks.append(i)
        else:
            imgs.append(img)
        all_imgs.append((img, img_original))
    imgs = np.array(imgs, dtype=np.uint8)
    X = np.multiply(imgs.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    preds = sess.run(prediction_print, feed_dict={x_print: X, keep_prob_print: 1.0}).tolist()
    point_list = [{'number': pred} for pred in preds]
    for i in blanks:
        point_list.insert(i, {'number': 0, 'isBlank': True})
    if DEBUG:
        for i in range(len(point_list)):
            point = point_list[i]['number']
            if 'isBlank' in point_list[i].keys():
                point = str(point) + '_blank'
            save_debug_info('id{}_{}'.format(i, point), all_imgs[i][0], all_imgs[i][1])

    return jsonify({'success': True, 'point': point_list})


@app.route('/readAnswerRightWrongs', methods=['POST'])
def read_answer_right_wrongs():
    """
    recognize r/w in batch
    """
    images = request.json.get('images')
    line = request.json.get('line')
    # border不传默认为False
    border = request.json.get('border') if request.json.get('border') is not None else False
    # is_black不传默认为True
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else True
    imgs = []
    all_imgs = []
    blanks = []
    try:
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'right_wrong', False) for i in range(len(images))]
    except:
        global pool
        pool = ProcessPoolExecutor(max_workers=16)
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'right_wrong', False) for i in range(len(images))]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    for i, future in enumerate(futures):
        img, img_original = future.result()
        if is_blank(img):
            blanks.append(i)
        else:
            imgs.append(img)
        all_imgs.append((img, img_original))
    imgs = np.array(imgs, dtype=np.uint8)
    X = np.multiply(imgs.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    preds = sess.run(prediction_right_wrong, feed_dict={x: X, keep_prob: 1.0}).tolist()
    point_list = [{'isRight': int(pred)} for pred in preds]
    for i in blanks:
        point_list.insert(i, {'isRight': 0, 'isBlank': True})
    if DEBUG:
        for i in range(len(point_list)):
            point = point_list[i]['isRight']
            if 'isBlank' in point_list[i].keys():
                point = str(point) + '_blank'
            save_debug_info('id{}_{}'.format(i, point), all_imgs[i][0], all_imgs[i][1])

    return jsonify({'success': True, 'point': point_list})


@app.route('/readAnswerWrongWithOthers', methods=['POST'])
def read_answer_wrong_with_others():
    """
    r/w and other
    """
    images = request.json.get('images')
    line = request.json.get('line')
    wrong_list = request.json.get('wrongList')
    if not wrong_list:
        # 0=错,1=对,2=半对,3=\\,4=/,5=其它,6=圈
        wrong_list = "0,2,6"
    # border不传默认为False
    border = request.json.get('border') if request.json.get('border') is not None else False
    # is_black不传默认为True
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else True
    imgs = []
    all_imgs = []
    blanks = []
    try:
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'right_wrong', False) for i in range(len(images))]
    except:
        global pool
        pool = ProcessPoolExecutor(max_workers=16)
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'right_wrong', False) for i in range(len(images))]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    for i, future in enumerate(futures):
        img, img_original = future.result()
        if is_blank(img):
            blanks.append(i)
        else:
            imgs.append(img)
        all_imgs.append((img, img_original))
    imgs = np.array(imgs, dtype=np.uint8)
    X = np.multiply(imgs.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    preds = sess.run(prediction_right_wrong_with_others, feed_dict={x_right_wrong_others: X, keep_prob_right_wrong_others: 1.0}).tolist()
    point_list = []
    is_other_list = []
    for pred in preds:
        is_other = 0 if str(pred) in wrong_list else 1
        is_other_list.append({'isOther': is_other})
        point_list.append({'isOther': pred})
    for i in blanks:
        point_list.insert(i, {'isOther': 0, 'isBlank': True})
        is_other_list.insert(i, {'isOther': 0, 'isBlank': True})
    if DEBUG:
        for i in range(len(point_list)):
            point = point_list[i]['isOther']
            if 'isBlank' in point_list[i].keys():
                point = str(point) + '_blank'
            save_debug_info('id{}_{}'.format(i, point), all_imgs[i][0], all_imgs[i][1])

    return jsonify({'success': True, 'point': is_other_list})


@app.route('/readAnswerOptions', methods=['POST'])
def read_answer_options():
    """
    Recognize letters and other
    """
    images = request.json.get('images')
    line = request.json.get('line')
    # border不传默认为False
    border = request.json.get('border') if request.json.get('border') is not None else False
    # is_black不传默认为True
    is_black = request.json.get('isBlack') if request.json.get('isBlack') is not None else True
    # is_black不传默认为False
    with_efg = request.json.get('withEFG') if request.json.get('withEFG') is not None else False
    imgs = []
    all_imgs = []
    blanks = []
    try:
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'abcd', True) for i in range(len(images))]
    except:
        global pool
        pool = ProcessPoolExecutor(max_workers=16)
        futures = [pool.submit(img_base64_to_array, images[i], border, line, is_black, 'abcd', True) for i in range(len(images))]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    for i, future in enumerate(futures):
        img, img_original = future.result()
        if is_blank(img):
            blanks.append(i)
        else:
            imgs.append(img)
        all_imgs.append((img, img_original))
    imgs = np.array(imgs, dtype=np.uint8)
    X = np.multiply(imgs.astype(np.float32), 1.0 / 255.0).reshape(-1, 784)
    preds = sess.run(prediction_abcdefg, feed_dict={x_abcdefg: X, keep_prob_abcdefg: 1.0}).tolist() if with_efg\
                                            else sess.run(prediction_abcd, feed_dict={x_abcd: X, keep_prob_abcd: 1.0}).tolist()
    point_list = [{'option': pred} for pred in preds]
    for i in blanks:
        point_list.insert(i, {'option': -1, 'isBlank': True})
    if DEBUG:
        for i in range(len(point_list)):
            point = point_list[i]['option']
            if 'isBlank' in point_list[i].keys():
                point = str(point) + '_blank'
            save_debug_info('id{}_{}'.format(i, point), all_imgs[i][0], all_imgs[i][1])

    return jsonify({'success': True, 'point': point_list})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
