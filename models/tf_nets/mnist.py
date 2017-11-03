import numpy as np
from dataset import Dataset, DSS
from tensorflow.python.framework import dtypes
import os

mnist_files = ['train-images-idx3-ubyte',
               'train-labels-idx1-ubyte',
               't10k-images-idx3-ubyte',
               't10k-labels-idx1-ubyte']

img_header_type = np.dtype([('magic', '>i4'),
                            ('num_images', '>i4'),
                            ('rows', '>i4'),
                            ('cols', '>i4')])
label_header_type = np.dtype([('magic', '>i4'),
                              ('num_items', '>i4')])
img_type = np.dtype('(28,28,1)B') # row, col, depth
label_type = np.dtype('B')

def dense_to_onehot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot

def extract_images(fobject):
    header = np.fromfile(fobject, dtype=img_header_type, count=1)
    images = np.fromfile(fobject, dtype=img_type)
    return header, images

def extract_labels(fobject, one_hot=True, num_classes=10):
    header = np.fromfile(fobject, dtype=label_header_type, count=1)
    labels = np.fromfile(fobject, dtype=label_type)
    if one_hot:
        lables = dense_to_onehot(labels, num_classes)
    return header, labels

def read_mnist(loc='/mnt/mnist'):
    for file_name in mnist_files:
        error_msg = file_name + ' not found!'
        assert os.path.exists(os.path.join(loc,file_name)), error_msg
    train_imgs_f = os.path.join(loc, mnist_files[0])
    train_labels_f = os.path.join(loc, mnist_files[1])
    test_imgs_f = os.path.join(loc, mnist_files[2])
    test_labels_f = os.path.join(loc, mnist_files[3])
    # Read training images
    with open(train_imgs_f, 'rb') as f:
        train_iheader, train_imgs = extract_images(f)
        assert train_iheader['magic'] == 2051
        assert train_iheader['num_images'] == 60000
        assert train_iheader['rows'] == 28
        assert train_iheader['cols'] == 28
    # Read training labels
    with open(train_labels_f, 'rb') as f:
        train_lheader, train_labels = extract_labels(f)
        assert train_lheader['magic'] == 2049
        assert train_lheader['num_items'] == 60000
    # Read testing images
    with open(test_imgs_f, 'rb') as f:
        test_iheader, test_imgs = extract_images(f)
        assert test_iheader['magic'] == 2051
        assert test_iheader['num_images'] == 10000
        assert test_iheader['rows'] == 28
        assert test_iheader['cols'] == 28
    # Read testing labels
    with open(test_labels_f, 'rb') as f:
        test_lheader, test_labels = extract_labels(f)
        assert test_lheader['magic'] == 2049
        assert test_lheader['num_items'] == 10000
    return train_imgs, train_labels, test_imgs, test_labels

def load_mnist(loc='/mnt/mnist', one_hot=True, dtype=dtypes.float32,
               val_size=5000, reshape=True):
    train_x, train_y, test_x, test_y = read_mnist(loc)
    assert 0 <= val_size <= len(train_x), "Invalid validation size!"
    if one_hot:
        train_y = dense_to_onehot(train_y, 10)
        test_y = dense_to_onehot(test_y, 10)
    val_x = train_x[:val_size]
    val_y = train_y[:val_size]
    train_x = train_x[val_size:]
    train_y = train_y[val_size:]
    options = dict(dtype=dtype, reshape=reshape)
    train = Dataset(train_x, train_y, **options)
    val = Dataset(val_x, val_y, **options)
    test = Dataset(test_x, test_y, **options)
    return DSS(train=train, val=val, test=test)

def pretty_print(img):
    # Util to quickly see img on console
    s = ""
    for row in img:
        s += "".join([str(pixel).zfill(3) for pixel in row])
        s += '\n'
    print(s)
