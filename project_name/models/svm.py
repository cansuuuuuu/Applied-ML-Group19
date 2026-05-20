from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure

IMG_SIZE = 64  # we set a small image size so baseline model can work well
PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "dataset"
TRAIN_DIR = DATASET / "train"
TEST_DIR = DATASET / "test"


def image_files(root):
    """
    function returns all images in the root directory.
    :param root: the root path of images.
    :return: sorted list of files that are images
    """
    return sorted(path for path in Path(root).iterdir() if path.is_file())


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
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        images = image_files(class_dir)
        for img_path in images:
            # convert to grayscale and resize
            img = Image.open(img_path).convert("L").resize((
                IMG_SIZE,
                IMG_SIZE))
            X.append(np.array(img))  # the image
            y.append(label)  # the label of the image
        print(f"{cls}: {len(images)} images")
    return np.array(X), np.array(y), classes


def flatten_pixels(X_train, X_test):
    """
    function for flattening to pixels.
    this is the feature we can use instead of HOG if we chose to.
    :param X_train: training set.
    :param X_test: test set.
    :return: the training and test set, flattened to pixels.
    """
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0
    return X_train, X_test


def hoggify(imgs):
    """
    function for changing the images to Histogram of Oriented Gradients (HOG).
    :param imgs: the images to be converted (training and test set).
    :return: the images converted to HOG.
    """
    hogged = np.array([
        hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        for img in imgs
    ])
    return hogged


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


def extract_features(X_train, X_test):
    """
    extracting features from images.
    :param X_train: the training set.
    :param X_test: the test set.
    :return: the training and test set, hoggified.
    """
    X_train_hog = hoggify(X_train)
    X_test_hog = hoggify(X_test)
    return X_train_hog, X_test_hog


def scale_features(X_train, X_test):
    """
    scaling the features.
    :param X_train: the training set.
    :param X_test: the test set.
    :return: the training and test set, scaled.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


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
    :return: the training and test set, scaled.
    """
    predictions = model.predict(X_test)
    print("SVM:\n", classification_report(
        y_test,
        predictions,
        target_names=classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


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
    X_train, X_test = extract_features(X_train, X_test)
    X_train, X_test, _ = scale_features(X_train, X_test)
    svm = train_svm(X_train, y_train)
    evaluate_model(svm, X_test, y_test, classes)
    show_hog_example(TRAIN_DIR/"1_cumulus/1_cumulus_000009.jpg")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()