import numpy as np
from dataset import Dataset, DSS
import os

mnist_files = ['train-images-idx3-ubyte',
               'train-labels-idx1-ubtye',
               't10k-images-idx3-ubyte',
               't10k-labels-idx1-ubtye']

img_header_type = np.dtype([('magic', '>i4'),
                            ('num_images', '>i4'),
                            ('rows', '>i4'),
                            ('cols', '>i4')])
label_header_type = np.dtype([('magic', '>i4'),
                              ('num_items', '>i4')])
img_type = np.dtype('(28,28)B')
label_type = np.dtype('B')

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_images(fobject):
    header = np.fromfile(fobject, dtype=img_header_type, count=1)
    images = np.fromfile(fobject, dtype=img_type)
    return header, images

def extract_labels(fobject, one_hot=True, num_classes=10):
    header = np.fromfile(fobject, dtype=label_header_type, count=1)
    labels = np.fromfile(fobject, dtype=label_type)
    return header, labels


def read_mnist(loc='/mnt/mnist'):
    for file_name in mnist_files:
        error_msg = file_name + ' not found!'
        assert os.path.exists(loc+file_name), error_msg
    train_imgs_f = loc + mnist_files[0]
    train_labels_f = loc + mnist_files[1]
    test_imgs_f = loc + mnist_files[2]
    test_labels_f = loc + mnist_files[3]
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
        train_iheader, train_imgs = extract_images(f)
        assert test_iheader['magic'] == 2051
        assert test_iheader['num_images'] == 10000
        assert test_iheader['rows'] == 28
        assert test_iheader['cols'] == 28
    # Read testing labels
    with open(test_labels_f, 'rb') as f:
        test_lheader, test_labels = extract_labels(f)
        assert test_lheader['magic'] == 2049
        assert test_lheader['num_items'] == 10000




    def __init__(self, loc='/mnt/mnist'):
        for file_name in mnist_files:
            error_msg = file_name + " not found!"
    	    assert os.path.exists(loc+file_name), error_msg
        self.train_imgs = loc + mnist_files[0]
        self.train_labels = loc + mnist_files[1]
        self.test_imgs = loc + mnist_files[2]
        self.test_labels = loc + mnist_files[3]

     
