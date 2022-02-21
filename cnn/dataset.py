#coding: utf-8
import struct
import numpy as np
import cv2
import gzip
from path import Path
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from tensorflow.python.framework import random_seed

from processimg import apply_middlewares


class DataSet:
    def __init__(self, images, labels, seed=None, use_thread=False, middlewares=[]):
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._shape = images.shape[1:]
        self._use_thread = use_thread
        self._middlewares = middlewares
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._pool = ThreadPoolExecutor()

    def __del__(self):
        self._pool.shutdown(wait=False)
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def shape(self):
        return self._shape

    @property
    def use_thread(self):
        return self._use_thread

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self.images[start:self._num_examples]
            labels_rest_part = self.labels[start:self._num_examples]
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            images_batch = np.concatenate((images_rest_part, images_new_part), axis=0).copy()
            labels_batch = np.concatenate((labels_rest_part, labels_new_part), axis=0).copy()
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            images_batch = self.images[start:end].copy()
            labels_batch = self.labels[start:end].copy()
        if self._middlewares:
            if not self.use_thread:
                # single thread
                for img in images_batch:
                    apply_middlewares(img, self._middlewares)
            else:
                # multi thread
                futures = {self._pool.submit(apply_middlewares, img) for img in images_batch}
                concurrent.futures.wait(futures, return_when=ALL_COMPLETED)

        images_batch = np.multiply(images_batch.astype(np.float32), 1.0 / 255.0)
        return images_batch.reshape(-1, self.shape[0]*self.shape[1]), labels_batch


def write_mnist_format_image_file(file_name, np_array, magic_number, length, shape, zip=True):
    op = gzip.open if zip else open
    with op(file_name, 'wb') as f:
        f.write(struct.pack('>IIII', magic_number, length, *shape))
        f.write(np_array.tobytes())
    
    
def write_mnist_format_label_file(file_name, np_array, magic_number, length, zip=True):
    op = gzip.open if zip else open
    with op(file_name, 'wb') as f:
        f.write(struct.pack('>II', magic_number, length))
        f.write(np_array.tobytes())
    
    
def read_mnist_format_image_file(file_name, zip=True, shuffle=False, seed=np.pi):
    op = gzip.open if zip else open
    _, length, x, y = np.array(struct.unpack('>IIII', op(file_name, 'rb').read(16)))
    images = np.array(struct.unpack_from('B'*length*x*y, op(file_name, 'rb').read(), 16), dtype=np.uint8).reshape(length, x, y)
    if shuffle:
        random.Random(seed).shuffle(images)
    return images


def read_mnist_format_label_file(file_name, label_length=10, one_hot=True, zip=True, shuffle=False, seed=np.pi):
    op = gzip.open if zip else open
    _, length = np.array(struct.unpack('>II', op(file_name, 'rb').read(8)))
    if one_hot:
        labels = np.zeros((length, label_length), dtype=np.uint8)
    else:
        labels = np.zeros((length), dtype=np.uint8)
    for i, label in enumerate(struct.unpack_from('B'*length, op(file_name, 'rb').read(), 8)):
        if one_hot:
            labels[i][label] = 1
        else:
            labels[i] = label
    if shuffle:
        random.Random(seed).shuffle(labels)
    return labels