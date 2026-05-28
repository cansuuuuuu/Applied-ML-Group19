from sklearn.model_selection import train_test_split
import numpy as np


def split(images, labels, val_size=0.2):
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=val_size, stratify=labels,random_state=42)
    return train_images, val_images, train_labels, val_labels



