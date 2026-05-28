from os import path

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data" / "dataset"
TRAIN_DIR = DATASET /"train"
TEST_DIR = DATASET / "test"

def image_files(root):
    """
    function returns all images in the root directory.
    :param root: the root path of images.
    :return: sorted list of files that are images
    """
    return sorted(p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def get_classes(root):
    """
    function returns classes in the directory based on the filenames.
    :param root: the root path of images.
    :return: sorted list of classes based on the subfolders.
    """
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


def load_images(root, classes=None):
    """
    function for loading in images from the dataset.
    :param root: the root path of images.
    :param classes: class folder names.
    :return: 3 arrays, one for images, one for labels, and of classes.
    """
    X, y = [], []
    classes = classes or get_classes(root)
    for label, cls in enumerate(classes):
        class_dir = Path(root) / cls
        images = image_files(class_dir)
        for img_path in images:
            # convert to BGR for feature extraction, and resize
            img = (Image.open(img_path).convert("RGB"))
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            X.append(arr)  # the image
            y.append(label)  # the label of the image
        print(f"{cls}: {len(images)} images")
    return np.array(X), np.array(y), classes


def load_dataset(train_dir=TRAIN_DIR, test_dir=TEST_DIR):
    """
    function loading in dataset. since the dataset is already split
    as train and test we can load images separately.
    :param train_dir: directory of training set.
    :param test_dir: directory of test set.
    :return: the training set, test set, and classes.
    """
    X_train, y_train, classes = load_images(train_dir)
    X_test, y_test, _ = load_images(test_dir, classes)
    return X_train, y_train, X_test, y_test, classes
