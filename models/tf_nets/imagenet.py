import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from scipy.misc import imread

rseed = 69

# On GTX1080 sever. Data is converted to 227x227

DATALOC = "/mnt/imagenet"
TRAIN_LOC = "train"
VAL_LOC = "val"
TRAIN_FILE = "train.txt"
VAL_FILE = "val.txt"
WORD_FILE = "synset_words.txt"
LABELS_FILE = "synsets.txt"

word_dict = {}  # Synset ids -> words. E.g. n01514668->cock
ids_list = []  # Label ids -> synset ids. E.g. 7->n01514668
train_data = None
val_data = None
train_size = 1281167
val_size = 50000

def strip(string):
    return string.strip()

def create_word_dict():
    global word_dict
    if len(word_dict) != 1000:
        word_file_path = os.path.join(DATALOC, WORD_FILE)
        with open(word_file_path) as f:
            for line in map(strip, f.readlines()):
                ids, words = line.split(' ', 1)
                word_dict[ids] = words
    assert len(word_dict) == 1000, "Something wrong with ids->words!"
    return word_dict
    
def create_ids_list():
    global ids_list
    if len(ids_list) != 1000:
        ids_file_path = os.path.join(DATALOC, LABELS_FILE)
        with open(ids_file_path) as f:
            for line in map(strip, f.readlines()):
                ids_list.append(line)
    return ids_list

def read_data():
    global train_data, val_data, train_size, val_size
    with open(os.path.join(DATALOC, TRAIN_FILE)) as f:
        train_data = [d for d in map(strip, f.readlines())]
        train_size = len(train_data)
    with open(os.path.join(DATALOC, VAL_FILE)) as f:
        val_data = [d for d in map(strip, f.readlines())]
        val_size = len(val_data)

def gen_data(batch_size, phase):
    global train_data
    global val_data
    if train_data is None or val_data is None:
        read_data()
    if phase.lower() == "train":
        data = train_data
        folder = os.path.join(DATALOC, TRAIN_LOC)
    else:
        data = val_data
        folder = os.path.join(DATALOC, VAL_LOC)
    num_batches = len(data)//batch_size
    # num_pad = len(data) - num_batches * batch_size
    while(True):
        data = shuffle(data)
        for i in range(num_batches):
            yield to_images_labels(data[i*batch_size:(i+1)*batch_size], folder)

def to_images_labels(data_lines, folder):
    """Note: channel last for cuDNN""" 
    images = []
    labels = []
    for datum in data_lines:
        image_path, label = datum.split(" ", 1)
        img = imread(os.path.join(folder, image_path), mode="RGB")
        img = img - np.mean(img)
        img = img / 255
        label = int(label)
        images.append(img)
        labels.append(label)
    return images, labels
