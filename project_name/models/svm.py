# from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             jaccard_score,
                             balanced_accuracy_score,
                             f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure
from project_name.features.feature_extraction import FeatureExtraction
from project_name.data.loading import load_dataset, split
from project_name.data.preprocessing import resize, normalise, augmenting_classes


# IMG_SIZE = 64  # we set a small image size so baseline model can work well
# PROJECT = Path(__file__).resolve().parents[1]
# DATASET = PROJECT / "data" / "dataset"
# TRAIN_DIR = DATASET / "train"
# TEST_DIR = DATASET / "test"


# def image_files(root):
#     """
#     function returns all images in the root directory.
#     :param root: the root path of images.
#     :return: sorted list of files that are images
#     """
#     return sorted(path for path in Path(root).iterdir() if path.is_file())


# def get_classes(root):
#     """
#     function returns classes in the directory based on the filenames.
#     :param root: the root path of images.
#     :return: sorted list of classes based on the subfolders.
#     """
#     return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


# def load_images(root, classes=None):
#     """
#     function for loading in images from the dataset.
#     :param root: the root path of images.
#     :param classes: class folder names.
#     :return: 3 arrays, one for images, one for labels, and of classes.
#     """
#     X, y = [], []
#     classes = classes or get_classes(root)
#     for label, cls in enumerate(classes):
#         class_dir = Path(root) / cls
#         images = image_files(class_dir)
#         for img_path in images:
#             # convert to BGR for feature extraction, and resize
#             img = (Image.open(img_path).convert("RGB").
#                    resize((IMG_SIZE, IMG_SIZE)))
#             arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             X.append(arr)  # the image
#             y.append(label)  # the label of the image
#         print(f"{cls}: {len(images)} images")
#     return np.array(X), np.array(y), classes


def flatten_pixels(X_train, X_test):
    """
    function for flattening to pixels.
    this is the feature we can use instead of HOG if we chose to.
    :param X_train: training set.
    :param X_test: test set.
    :return: the training and test set, flattened to pixels.
    """

    # convert the images to grayscale
    # for flattening pixels
    def to_gray(arr):
        if arr.ndim == 4:
            return np.array([cv2.cvtColor(
                im,
                cv2.COLOR_BGR2GRAY) for im in arr])
        return arr

    X_train = to_gray(X_train).reshape(
        len(X_train), -1).astype(np.float32) / 255.0
    X_test = to_gray(X_test).reshape(
        len(X_test), -1).astype(np.float32) / 255.0

    return X_train, X_test


def hoggify(imgs):
    """
    function for changing the images to
    Histogram of Oriented Gradients (HOG).
    :param imgs: the images to be converted (training and test set).
    :return: the images converted to HOG.
    """
    hogged = np.array([
        hog(
            # convert to grayscale
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        for img in imgs
    ])
    return hogged


# def load_dataset(train_dir=TRAIN_DIR, test_dir=TEST_DIR):
#     """
#     function loading in dataset. since the dataset is already split
#     as train and test we can load images separately.
#     :param train_dir: directory of training set.
#     :param test_dir: directory of test set.
#     :return: the training set, test set, and classes.
#     """
#     X_train, y_train, classes = load_images(train_dir)
#     X_test, y_test, _ = load_images(test_dir, classes)
#     return X_train, y_train, X_test, y_test, classes


def extract_features(X_train, X_val, X_test):
    """
    extracting features from images. here the choice could
    be made between HOG, flattening pixels, or the feature
    extraction class we made based on the research paper
    :param X_train: the training set.
    :param X_test: the test set.
    :return: the training and test set.
    """
    """uncomment the following lines to use HOG"""
    # X_train = hoggify(X_train)
    # X_test = hoggify(X_test) 
    # X_val = hoggify(X_val)

    """uncomment the following line to flatten pixels"""
    # X_train, X_val, X_test = flatten_pixels(X_train, X_val, X_test)

    feature_extractor = FeatureExtraction()
    X_train_feats = np.array(
        [list(feature_extractor.run(img).values()) for img in X_train])
    X_val_feats = np.array(
        [list(feature_extractor.run(img).values()) for img in X_val])
    X_test_feats = np.array(
        [list(feature_extractor.run(img).values()) for img in X_test])

    return X_train_feats, X_val_feats, X_test_feats


def scale_features(X_train,X_val, X_test):
    """
    scaling the features.
    :param X_train: the training set.
    :param X_val: the validation set.
    :param X_test: the test set.
    :return: the training and test set, scaled.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_svm(X_train, y_train):
    """
    trains the SVM.
    :param X_train: the training set.
    :param X_test: the test set.
    :return: the trained SVM model.
    """
    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train, y_train)
    return svm


def evaluate_model(model, X_test, y_test, classes):
    """
    evaluates the model
    :param model: the trained SVM model.
    :param X_test: the test set.
    :param y_test: the test labels.
    :param classes: the classes of the test set.
    :return: the evaluation metrics of the model.
    """
    predictions = model.predict(X_test)
    print("SVM:\n", classification_report(
        y_test,
        predictions,
        target_names=classes
    ))
    print("Confusion Matrix:\n",
          confusion_matrix(y_test, predictions))
    # metrics 4 imbalance in the dataset
    print("Jaccard Similarity: ",
          jaccard_score(y_test, predictions, average='macro'), "\n")
    print("Balanced Accuracy: ",
          balanced_accuracy_score(y_test, predictions), "\n")
    print("Macro F1 Score: ",
          f1_score(y_test, predictions, average='macro'), "\n")


def show_hog_example(image_path):
    """
    shows an example of how Histogram of Oriented Gradients (HOG) looks.
    :param image_path: a path to an image to use as an example.
    :return: nothing.
    """
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    _, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        sharex=True,
        sharey=True)

    ax1.axis("off")
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title("Input image")

    hog_image_rescaled = exposure.rescale_intensity(
        hog_image,
        in_range=(0, 10))

    ax2.axis("off")
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title("Histogram of Oriented Gradients")
    plt.show()


def run_pipeline():
    """
    runs the SVM pipeline
    :param: nothing.
    :return: nothing.
    """
    X_train, y_train, X_test, y_test, classes = load_dataset()
    X_train, X_val, y_train, y_val = split(X_train, y_train)
    X_train = resize(X_train)
    X_val = resize(X_val)
    X_test = resize(X_test)
    
    X_train = normalise(X_train)
    X_val = normalise(X_val)
    X_test = normalise(X_test)
    
    # X_train, y_train = augmenting_classes(X_train, y_train)

    X_train, X_val, X_test = extract_features(X_train, X_val, X_test)
    X_train, X_val, X_test, _ = scale_features(X_train, X_val, X_test)
    svm = train_svm(X_train, y_train)
    evaluate_model(svm, X_val, y_val, classes)
    evaluate_model(svm, X_test, y_test, classes)
    # show_hog_example(TRAIN_DIR/"1_cumulus/1_cumulus_000009.jpg")


# def main():
#     run_pipeline()


# if __name__ == "__main__":
#     main()
