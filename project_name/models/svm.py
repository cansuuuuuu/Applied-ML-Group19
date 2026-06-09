# from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             jaccard_score,
                             balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import exposure
from project_name.features.feature_extraction import FeatureExtraction
from sklearn.feature_extraction._dict_vectorizer import DictVectorizer
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




def scale_features(X_train, X_test):
    """
    scaling the features.
    :param X_train: the training set.
    :param X_val: the validation set.
    :param X_test: the test set.
    :return: the training and test set, scaled.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def build_svm_pipeline() -> GridSearchCV:
    """
    Build the pipeline for the SVM training grid search
    :param None
    :return: the grid search cv pipeline for the SVM
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("svm", SVC(kernel = "rbf"))
    ])

    parameter_grid = {
        'pca_n_components' : [0.8, 0.85, 0.9, 0.95, 0.99, None],
        'svm_C' : [0.1, 1, 10, 100],
        'svm_gamma' : ["scale", 0.001, 0.01, 0.1],
    }

    cross_validation = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    grid_search = GridSearchCV( estimator = pipeline,
                         param_grid = parameter_grid,
                         scoring = {
                        "balanced_accuracy": "balanced_accuracy",
                        "macro_f1": "f1_macro"
                        },
                         refit="macro_f1",
                         cv = cross_validation,
                         n_jobs = -1,
                         verbose = 1,
                         return_train_score = True
                         )

    return grid_search

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


    X_train = resize(X_train)
    X_test = resize(X_test)
    
    X_train = normalise(X_train)
    X_test = normalise(X_test)

    feature_extractor = FeatureExtraction()

    train_dicts = [feature_extractor.run(img) for img in X_train]
    test_dicts = [feature_extractor.run(img) for img in X_test]

    vectorizer = DictVectorizer(sparse = False)

    X_train_feats = vectorizer.fit_transform(train_dicts)
    X_test_feats = vectorizer.transform(test_dicts)

    grid = build_svm_pipeline()
    grid.fit(X_train_feats, y_train)

    best_model = grid.best_estimator_

    joblib.dump(best_model, "project_name/models/svm_model.pkl")
    vectorizer = vectorizer.get_feature_names_out()
    joblib.dump(vectorizer, "project_name/models/vectorizer.pkl")



    evaluate_model(best_model, X_test_feats, y_test, classes)
    # show_hog_example(TRAIN_DIR/"1_cumulus/1_cumulus_000009.jpg")


def main():
     run_pipeline()


if __name__ == "__main__":
     main()
