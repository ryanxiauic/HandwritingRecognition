# -*- coding: utf-8 -*-
import sys
import random
from random import choice, random
import cv2
import numpy as np
from path import Path
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from processimg import combine_images, apply_middlewares, draw_word
from dataset import read_mnist_format_image_file, read_mnist_format_label_file


def read_imgs_file(image_file_path, label_file_path, label_length, one_hot=True, use_thread=False, middlewares=[], 
                   additional_fonts=[], additional_fonts_each_group=0, additional_sample_path=None):
    """
    :description: read imgs from file
    :param image_file_path: image path
    :param label_file_path: label path
    :param label_length: label length
    :param one_hot: if label as one hot encoding
    :param use_thread: use thread or not
    :param middlewares: read img while process with coefficients
    :param additional_fonts: compatible with printing font
    :param additional_fonts_each_group: the fonts num
    :param additional_sample_path: additional sample path
    :return ndarray格式的imgs, labels
    """
    imgs = read_mnist_format_image_file(image_file_path)
    labels = read_mnist_format_label_file(label_file_path, label_length, one_hot=one_hot)
    _, x, y = imgs.shape
    if len(additional_fonts) != 0 and label_length == 10:
        print('before printer font', imgs.shape)
        add_length = label_length * len(additional_fonts) * additional_fonts_each_group
        add_imgs = np.zeros((add_length, x, y), dtype='uint8')
        add_labels = np.zeros((add_length, label_length), dtype='uint8')
        i = 0
        for group in range(additional_fonts_each_group):
            for number in range(label_length):
                for font_path in additional_fonts:
                    add_imgs[i] = draw_word(str(number), font_path)
                    add_labels[i][number] = 1
                    i += 1
        imgs = np.concatenate((imgs, add_imgs), axis=0)
        labels = np.concatenate((labels, add_labels), axis=0)
        print('after printer font', imgs.shape)
    
    if additional_sample_path is not None:
        print('before add sample', imgs.shape)
        add_imgs, add_labels = read_imgs_dir(additional_sample_path, (x, y), label_length, one_hot=True)
        imgs = np.concatenate((imgs, add_imgs), axis=0)
        labels = np.concatenate((labels, add_labels), axis=0)
        print('after add sample', imgs.shape)
    _apply_middlewares_for_all_image(imgs, middlewares, use_thread=use_thread)
    return imgs, labels


def read_imgs_file_by_group(image_file_path, label_file_path, label_length, groups, 
                            other=None, other_count=1, 
                            last_merge_other=False, other_probability=0.5, 
                            use_thread=False, middlewares=[], additional_sample_path=None):
    """
    :description: read imgs by group
        return num of images：groups * label_length

    :param image_file_path: image path
    :param label_file_path: label path
    :param label_length: label num
    :param groups: total sample size=label_length*groups
    :param other: the function to generate other, if necessary
    :param other_count: num of other label
    :param last_merge_other: For recognize right or wrong, regard wrong as 0, right and othere as 1
    :param other_probability: Same as last parameter, the probability of other
    :param use_thread: thread or not
    :param middlewares: coefficients to process imgs
    :param additional_sample_path: additional sample path
    :return ndarray of imgs, labels
    """
    if other_count != 1 and last_merge_other == True:
        raise ValueError('last merge other only support 1 other count')
    _imgs = read_mnist_format_image_file(image_file_path)
    _, x, y = _imgs.shape
    _labels = read_mnist_format_label_file(label_file_path, label_length)
    sample_cnt = groups * label_length
    label_dict = defaultdict(list)
    imgs = np.zeros((sample_cnt, x, y), dtype='uint8')
    labels = np.zeros((sample_cnt, label_length), dtype='uint8')

    if additional_sample_path is not None:
        print('before add sample', _imgs.shape)
        add_imgs, add_labels = read_imgs_dir(additional_sample_path, (x, y), label_length, one_hot=True)
        _imgs = np.concatenate((_imgs, add_imgs), axis=0)
        _labels = np.concatenate((_labels, add_labels), axis=0)
        print('after add sample', _imgs.shape)
        
    for i, label in enumerate(_labels):
        label_dict[tuple(label).index(1)].append(i)
    for i in range(groups):
        for j in range(label_length):
            other_flag = False
            # others 标签图片随机生成，随机生成的图片的Label值也是others中的任意一个
            if j >= label_length - other_count and other:
                if not last_merge_other:
                    other_flag = True
                elif last_merge_other and random() < other_probability: 
                    other_flag = True
            if other_flag:
                imgs[i*label_length+j:(i+1)*label_length] = other()
                l = 0
                for k in range(i*label_length+j, (i+1)*label_length):
                    labels[k][j+l] = 1
                    l += 1
                break
            else:
                if j < 10:
                    imgs[i*label_length+j] = _imgs[choice(label_dict[j])]
                else:
                    img1 = _imgs[choice(label_dict[int(str(j)[0])])]
                    img2 = _imgs[choice(label_dict[int(str(j)[1])])]
                    imgs[i*label_length+j] = combine_images([img1, img2])
                labels[i*label_length+j][j] = 1
        sys.stdout.write("[%d/%d] loading \r" % (i+1, groups))
        sys.stdout.flush()

    _apply_middlewares_for_all_image(imgs, middlewares, use_thread=use_thread)
    return imgs, labels


def read_imgs_dir(img_dir, shape, label_length=10, one_hot=True):
    """
    :description: read imgs and labels
    :param img_dir: the structure should be like
        img_dir/
            0_xxx/              # 0 is a label
                xxxx.png        # the img should be processed using main.process_img
                xxxx.png
                ...
                xxxx.png
            1_xxx/
                xxxx.png
                xxxx.png
                ...
                xxxx.png
            ...
    :param shape: size of img
    :param label_length: label num
    :param one_hot:  if label is one-hot encoding
    :return ndarray of imgs, labels
    """
    x, y = shape
    sample_cnt = sum([len(d.files()) for d in Path(img_dir).dirs()])
    imgs = np.zeros((sample_cnt, x, y), dtype='uint8')
    if one_hot:
        labels = np.zeros((sample_cnt, label_length), dtype='uint8')
    else:
        labels = np.zeros((sample_cnt), dtype='uint8')
    index = 0
    for _dir in Path(img_dir).dirs():
        dir_label = int(_dir.name[0])
        for img_path in _dir.files():
            imgs[index] = cv2.imread(img_path, 0)
            if one_hot:
                labels[index][dir_label] = 1
            else:
                labels[index] = dir_label
            index += 1
    return imgs, labels


def _apply_middlewares_for_all_image(imgs, middlewares, use_thread):
    """
    :description: process img in batch
    :param imgs: ndarray of img
    :param use_thread: use thread or not
    :param middlewares: coefficients to process imgs
    """
    if not use_thread:
        for img in imgs:
            apply_middlewares(img, middlewares)
    else:
        pool = ThreadPoolExecutor()
        futures = {pool.submit(apply_middlewares, img) for img in imgs}
        concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
        pool.shutdown(wait=True)
