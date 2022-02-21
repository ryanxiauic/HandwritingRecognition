#coding: utf-8
from __future__ import absolute_import
from random import randint, uniform, choice
from functools import partial, reduce
from itertools import chain, combinations
import skimage.measure
import cv2
import numpy as np
from imutils import rotate_bound
from path import Path
from PIL import Image, ImageDraw, ImageFont

from rectangle import Rectangle

# from IPython.display import Image   #Image(filename='redraw.gif') reaveal in jupyter


def process_img(img, border=False, line=None, is_black=True, img_type='digit', clear_noise=True):
    """
    :description: process image(remove border, clear noise , transform to format as mnist)
    :param img: image
    :param border: have border or not
    :param line: the coordinates of upper left and bottom right point, format should be like x1-y1-x2-y2
    :param is_black: black background or not
    :param img_type: image type('digit', 'right_wrong', 'abcd')
    :param clear_noise: clear noise or not
    :return: image after process
    """
    img = img.copy()
    if not is_black:
        # the img need to be binarization and reverted if the img's background is white 
        img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV)[1]
    if border:
        if line is None:
            # crop： up 16 down 12 left 13 right 15
            clip_size = 14
            shape_x, shape_y = img.shape
            img = img[clip_size+2: shape_y-clip_size+2, clip_size-1: shape_x-clip_size-1]
        else:
            x1, y1 ,x2, y2 = line.split('-')
            x1, y1 ,x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img = img[y1+2: y2-1, x1+2:x2-1]
    elif img_type != 'digit':
        k_size = max(1, max(*get_image_real_size(img)) // 23)  # 23 is experiential coefficient, means when the size increase 23 pixel, the erosion kernel increase 1 
        img = broaden_img(img, k_size)
        img = cv2.dilate(img, np.ones((k_size, k_size), dtype='uint8'), iterations=1)
        if img_type == 'abcd':
            max_distance = (8,2)
        else:
            max_distance = 2
        #去噪
        img = denoise(img, max_class=clear_noise, max_distance=max_distance, img_type=img_type)
    else:
        img = denoise(img, max_class=clear_noise, img_type=img_type)
        k_size = max(1, max(*img.shape) // 23)
        img = broaden_img(img, k_size)
        img = cv2.dilate(img, np.ones((k_size, k_size), dtype='uint8'), iterations=1)
    if len(np.transpose(img.nonzero())) <= 20:
        # if non-zero pixel less than 20, it will be reagarded as empty
        return np.zeros((28, 28), dtype='uint8')
    if img_type == 'right_wrong':
        img = to_mnist_image(img)
    else:
        img = get_min_width_image(img)
        img = to_mnist_image(img)
    return img


def broaden_img(img, size):
    """
    boraden image, fill with black pixels
    :param img: original image
    :param size: pixels to be padded
    :return: image after broaden
    """
    w,h = img.shape
    new_img = np.zeros((w+2*size, h+2*size))
    new_img[size:w+size, size:h+size] = img
    return new_img

def draw_word(word, font_path, font_size=75):
    """
    :description: draw words using specific font，resize to 28X28
    :param word: words
    :param font_path: font
    :param font_size: font size
    :return: image after process
    """
    font = ImageFont.truetype(font_path, font_size)
    width, height = font.getsize(word)
    _, (offset_x, offset_y) = font.font.getsize(word)
    pil_img = Image.new('L', (width, height-offset_y), color=0)
    draw = ImageDraw.Draw(pil_img)
    draw.text((-offset_x, -offset_y), word, font=font, fill=0xFFFFFF)
    return to_mnist_image(np.array(pil_img))


def random_draw_words(words, font_dir):
    """
    :description: random draw words with size of 28X28
    :param words: string
    :param font_dir: font dir
    :return: image after process
    """
    imgs = [draw_word(word, choice(Path(font_dir).files('*.ttf'))) for word in words]
    return combine_images(imgs, random_scale=0.1, random_rotate_degree=5, random_h_offset=0.2, random_v_offset=0.2)


def combine_images(imgs, random_scale=0.3, random_rotate_degree=15, random_h_offset=0.5, random_v_offset=0.3):
    """
    :description: combine images horizontally，and resize to 28X28
    :param imgs: images list
    :param random_scale: random scale ratio
    :param random_rotate_degree: radom rotate degree
    :param random_h_offset: horizontal offset
    :param random_v_offset: vertical offset
    :return: image after process
    """
    resized_imgs = []
    max_width = max_height = 0
    for img in imgs:
        img = center_image(img)
        if random_rotate_degree:
            img = random_rotate(img, (-random_rotate_degree, random_rotate_degree))
        img = center_image(img)
        shape_0, shape_1 = img.shape
        if random_scale:
            scale = uniform(1-random_scale, 1+random_scale)
            new_size = (int(np.round(shape_1 * scale)), int(np.round(shape_0 * scale)))
            resized_img = cv2.resize(img, new_size)
        else:
            resized_img = img
        resized_imgs.append(resized_img)
        
        max_width = max(max_width, resized_img.shape[1])
        max_height = max(max_height, resized_img.shape[0])
    max_width = min(max_width, 10)  # the width would be too small if number is 1
    max_h_offset = max(1, int(max_width*random_h_offset))
    max_v_offset = int(max_height*random_v_offset)
    
    new_imgs = []
    for i, img in enumerate(resized_imgs):
        h_offset = randint(1, max_h_offset)
        h_offset = 0 if i+1==len(imgs) else h_offset  # Not need to pad at right for last img
        v_offset = randint(-max_v_offset, max_v_offset)
        # After process, the image should be at the same height
        img = padding_with_offset(img, max_height, h_offset, v_offset)
        new_imgs.append(img)
    #return padding_image(np.hstack(new_imgs), border=2, resize=(28, 28))
    return to_mnist_image(np.hstack(new_imgs))


def padding_up_down(img, rows, is_up=False):
    """
    :description: pad at up or down
    :param img: 
    :param rows: 
    :param is_up: 
    :return: 
    """
    for _ in range(rows):
        if is_up:
            img = np.insert(img, 0, 0, axis=0)
        else:
            img = np.insert(img, img.shape[0], 0, axis=0)
    return img


def padding_right(img, columns):
    """
    :description: pad at right
    :param img: 
    :param rows: 
    :return: 
    """
    for _ in range(columns):
        img = np.insert(img, img.shape[1], 0, axis=1)
    return img
    
    
def padding_with_offset(img, target_height, h_offset, v_offset):
    """
    :description: pad at upper, bottom and right
    :param img: 
    :param target_height: 
    :param h_offset: 
    :param v_offset: 
    :return: 
    """
    # right
    img = padding_right(img, h_offset)
    # upper and bottom
    original_height = img.shape[0]
    if original_height < target_height:
        if original_height + abs(v_offset) >= target_height:
            # add target_height - original_height rows
            img = padding_up_down(img, target_height-original_height, v_offset>=0)
        else:
            # random add skewness
            img = padding_up_down(img, abs(v_offset), v_offset>=0)
            other_offset = target_height - original_height - abs(v_offset)
            down_offset = other_offset // 2
            up_offset = other_offset - down_offset
            img = padding_up_down(img, down_offset, False)
            img = padding_up_down(img, up_offset, True)
    return img


def apply_middlewares(img, middlewares=[]):
    """
    :description: process batch
    :param img: 
    :param middlewares: process with coefficients while reading
    """
    img[:] = reduce(lambda res, middleware: middleware(res), middlewares, img)
    

def random_rotate(img, degree_range=(-10, 10)):
    """
    :description: radom ratate
    :param img: 
    :param degree_range: 
    :return: 
    """
    return rotate_bound(img, uniform(*degree_range))


def random_erose(img, max_k=1):
    """
    :description: random erose
    :param img: 
    :param max_k: 
    :return: 
    """
    k = choice(range(max_k))
    return cv2.dilate(img, np.ones((k, k), dtype='uint8'))


def shear_affine_transform(img, x):
    """
    :description: Affine transform
    :param img: 
    :param x: 
    :return: 
    """
    if x == 0:
        return img
    M = [[1, x, 0], [0, 1, 0], [0, 0, 1]]
    nonzero = np.transpose(img.nonzero())
    new_nonzero = {}
    for point in nonzero:
        r = [point[1], point[0], 1]
        affine_point = np.matmul(M, r)
        new_point = (int(affine_point[1]), int(affine_point[0]))
        new_nonzero[new_point[0], new_point[1]] = img[point[0], point[1]]
                          
    _, (tl, br) = order_points(list(new_nonzero.keys()))
    shape_x = br[0] - tl[0] + 1
    shape_y = br[1] - tl[1] + 1
    new_img = np.zeros((shape_x, shape_y), dtype='uint8')
    for point, value in new_nonzero.items():
        new_img[point[0]-tl[0], point[1]-tl[1]] = value
    return new_img


def get_image_real_size(img):
    """
    :description: get real size after remove the black border
    :param img: 
    :return: 
    """

    if is_blank(img):
        return 0, 0
    nonzeros = np.transpose(np.nonzero(img))
    _, (tl, br) = order_points(nonzeros)
    return br[1]-tl[1], br[0]-tl[0]


def get_mass(img, axis):
    """
    :descripiton: get the mass of image
    :param img: 
    :param axis: 
    :return 
    """
    n = img.shape[axis]
    s = [1] * img.ndim
    s[axis] = -1
    i = np.arange(1, n+1).reshape(s)
    return np.sum(img * i) / np.sum(img)


get_mass_col = partial(get_mass, axis=1)
get_mass_row = partial(get_mass, axis=0)


def to_mnist_image(img):
    """
    :description: transform as the format of mnist, the description as following:
        The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box 
        while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing 
        technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the 
        center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.
    :param img: 
    :return: 
    """
    img = center_image(img)
    # First step，pad to the same height and width
    m, n = img.shape
    max_shape = max(img.shape)
    if m == n:
        padding_img = img.copy()
    else:
        padding_img = np.zeros((max_shape, max_shape), dtype='uint8')
        # centered
        d = abs(m - n) // 2
        if m > n:
            padding_img[:, d: n + d] = img
        else:
            padding_img[d: m + d, :] = img
    # Step 2，transform to 20 * 20
    if max_shape == 20:
        resized_img = padding_img
    else:
        resized_img = cv2.resize(padding_img, (20, 20), interpolation=cv2.INTER_CUBIC)
    # Step 3, pad to（28，28），the mass at the center
    target_shape = np.array((28, 28))
    real_shape = np.array(resized_img.shape) # 20 X 20
    # (center of mass)
    com = np.array((get_mass_row(resized_img), get_mass_col(resized_img)))
    before_center = (real_shape) / 2
    before_offset = com - before_center
    add_by_side = (target_shape - real_shape) / 2
    after_offset = add_by_side - before_offset
    after_offset = np.rint(after_offset).astype(int) + 1
    start_bound = after_offset.copy()
    if start_bound[0] < 0:
        after_offset[0] = 0
    if start_bound[1] < 0:
        after_offset[1] = 0
    end_bound = after_offset + real_shape - target_shape
    if end_bound[0] > 0:
        after_offset[0] -= end_bound[0]
    if end_bound[1] > 0:
        after_offset[1] -= end_bound[1]
    
    mnist_img = np.zeros(target_shape, dtype='uint8')
    mnist_img[after_offset[0]: after_offset[0]+real_shape[0], after_offset[1]: after_offset[1]+real_shape[1]] = resized_img
    return mnist_img


def get_min_width_image(img, k=10, step=0.1, debug=False):
    """
    :description: Shear affine transformation, and choose the img with smallest width
    :param img: 
    :param k: 
    :param step: 
    :param debug: 
    :return: 
    """
    assert k > 0
    img = center_image(img)
    img_min_width = img
    current_min_width, _ = get_image_real_size(img)
    if get_image_real_size(shear_affine_transform(img, step))[0] < current_min_width:
        r = range(1, k+1)
    elif get_image_real_size(shear_affine_transform(img, -step))[0] < current_min_width:
        r = range(-1, -k-1, -1)
    else:
        r = chain(range(1, k//2), range(-1, -k//2, -1))
    current_min_width -= 1 
    all_try_images = [img,]
    for i in r:
        M = i * step
        each = shear_affine_transform(img, M)
        width, _ = get_image_real_size(each)
        if width < current_min_width: 
            all_try_images.append(each)
            current_min_width = width
            img_min_width = each
    all_try_images.append(img_min_width)
    return (img_min_width, all_try_images) if debug else img_min_width


def is_special_5(img, rect_list):
    """
    :description: to check if the img is a special 5, which split in 2 parts
    :param img: 
    :param rect_list:
    :return: 
    """
    img_h, img_w = img.shape[:2]
    for rect in rect_list:
        # Rectangle(row_start, column_start, row_start + height,  column_start + width)
        rect_h, rect_w = rect.x2-rect.x1, rect.y2-rect.y1
        if rect.x2 <= 0.5 * img_h and rect.y2 >= 0.5 * img_w:
            if rect_w / rect_h > 1.7:
                return True
    return False


def denoise(img, max_class=True, max_distance=2, img_type='digit'):
    """
    :description: Denoise the img
    :param img: 
    :param max_class:  using the largest pattern as output
    :param max_distance: the max distance to combine patterns
    :param img_type: img type('digit', 'abcde', 'right_wrong')
    :return: img after process 
    """
    w, h = get_image_real_size(img)
    total_points = w * h
    classes = get_image_class(img)
    if img_type == 'abcde': #configuration for special E.
        rate = 0.0
    else:
        rate = 0.01
    #  remove patterns less than 1/100 of total pixels
    if rate != 0.0:
        classes = [clas for clas in classes if len(clas) > total_points * rate]
    if len(classes) == 0:
        return class_to_image(img, [])
    elif len(classes) == 1:
        return class_to_image(img, classes[0])
    while True:
        for i, j in combinations(range(len(classes)), 2):
            rect_i, rect_j = get_class_bounding_rect(classes[i]), get_class_bounding_rect(classes[j])
            if rect_i & (rect_j > max_distance) or (img_type == 'digit' and is_special_5(img, (rect_i, rect_j))):
                # combine for override pattern or special case
                classes[i].extend(classes[j])
                del classes[j]
                break
        else:
            break
    if max_class:
        max_clas = sorted(classes, key=lambda clas: -len(clas))[0]
        return class_to_image(img, max_clas)
    else:
        return classes_to_image(img, classes)


def get_image_class(img):
    """
    :description: Get pattern type
    :param img: 
    :return: 
    """
    img_0 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    label = skimage.measure.label(img_0, connectivity=2)
    rps = skimage.measure.regionprops(label)
    return [rp.coords.tolist() for rp in rps]


def class_to_image(img, clas):
    """
    :description: draw a pattern
    :param img: 
    :param clas: 
    :return: 
    """
    new_img = np.zeros_like(img)
    for row, column in clas:
        new_img[row][column] = img[row][column]
    return new_img


def classes_to_image(img, classes):
    """
    :description: draw patterns
    :param classes: 
    :return: 
    """
    new_img = np.zeros_like(img)
    for clas in classes:
        for row, column in clas:
            new_img[row][column] = img[row][column]
    return new_img


def get_class_bounding_rect(clas):
    """
    :description: get rectangle for given class
    :clas: 
    :return: 
    """
    row_start, column_start, height, width = cv2.boundingRect(np.array(clas))
    return Rectangle(row_start, column_start, row_start + height,  column_start + width)


def padding_image(img, border=0, resize=None):
    """
    :description: archived, replace with to_mnist_image
    :param img: 
    :param border: 
    :param resize: 
    :return: 
    """
    x, y = img.shape
    max_shape = max(img.shape)
    new_img = np.zeros((max_shape, max_shape), dtype='uint8')
    d = abs(x - y) // 2
    if x > y:
        new_img[:, d: y + d] = img
    elif x < y:
        new_img[d: x + d, ] = img
    else:
        new_img = img
    if border != 0 and resize is not None:
        new_size = (resize[0] - 2 * border, resize[1] - 2 * border)
    if resize is not None:
        resize_img = cv2.resize(new_img, new_size, interpolation=cv2.INTER_CUBIC)
        border_img = np.zeros(resize, dtype='uint8')
        border_img[border: border + new_size[0], border: border + new_size[1]] = resize_img
        return border_img

    if border != 0:
        border_img = np.zeros((max_shape + 2 * border, max_shape + 2 * border), dtype='uint8')
        border_img[border: border + max_shape, border: border + max_shape] = new_img
        return border_img
    return new_img


def center_image(img):
    """
    :description: remove black paddings
    :param img:
    :return: 
    """
    img = img.copy()
    _, ((t,l), (b, r)) = order_points(np.transpose(img.nonzero()))
    img = img[t: b + 1, l: r + 1]
    return img


def is_blank(img):
    """
    :description: check if the img is blank
    :param img: 
    :return: 
    """
    return not np.transpose(img.nonzero()).any()


def order_points(pts):
    """
    :description: Get coordinates of every non-empty pixels. For example
        s   O o o o O
            o o o o o
            o o o o o o o
        o o o o o o o o o 
            o o o o o o o
            o o o o o
            O o o o O   e
      For the img, o or O represent non-empty pixel (ignore s,e), then:
        1. 0 at the upper-left written as tl(top left)
        2. 0 at the upper-right written as tr(top right)
        3. The same for bl(bottom left)
        4. The same for br(bottom right)
      s/e is the upper-left/bottom-right of the whole img(s=start, e=end)
    :param pts: non-empty piexel coordinates, pts = np.transpose(img.nonzero())
    :return: tuple as ([tl, tr, br, bl], [start, end])
    """
    pts = np.array(pts).reshape((-1, 2))
    rect1 = np.zeros((4, 2), dtype="int32")
    rect2 = np.zeros((2, 2), dtype="int32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    bl = pts[np.argmin(diff)]
    tr = pts[np.argmax(diff)]
    
    min_x = np.argmin(pts[:, 0])
    min_y = np.argmin(pts[:, 1])
    start = (pts[min_x][0], pts[min_y][1])
    max_x = np.argmax(pts[:, 0])
    max_y = np.argmax(pts[:, 1])
    end = (pts[max_x][0], pts[max_y][1])
    return [tl, tr, br, bl], [start, end]