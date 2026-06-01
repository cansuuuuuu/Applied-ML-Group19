from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from PIL import Image

from project_name.models.cnn import CNNModel
from project_name.data.preprocessing import (
    build_train_dataset,
    build_val_dataset,
    IMAGE_SIZE,
)

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
#from project_name.models.svm import SVMModel


PROJECT = Path(__file__).resolve().parent / "project_name"
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
                64,
                64))
            X.append(np.array(img))  # the image
            y.append(label)  # the label of the image
        print(f"{cls}: {len(images)} images")
    return np.array(X), np.array(y), classes


def load_dataset(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
):
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


def gather_image_paths(root, classes=None):
    """Return lists of file paths and labels (ints) for a directory structured by class subfolders."""
    image_paths: list[str] = []
    labels: list[int] = []
    classes = classes or get_classes(root)
    for label, cls in enumerate(classes):
        class_dir = Path(root) / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        for p in files:
            image_paths.append(str(p))
            labels.append(label)
        print(f"{cls}: {len(files)} images")
    return image_paths, labels, classes


def save_dataset_examples(dataset, class_names, save_path, max_images=16):
    plt.figure(figsize=(10, 10))

    shown = 0
    for images, labels in dataset:
        for image, label in zip(images, labels):
            if shown >= max_images:
                plt.tight_layout()
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"Saved examples to {save_path}")
                return

            plt.subplot(4, 4, shown + 1)

            image = image.numpy()
            if image.shape[-1] == 1:
                plt.imshow(image.squeeze(), cmap="gray")
            else:
                plt.imshow(image)

            plt.title(class_names[int(label)])
            plt.axis("off")
            shown += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved examples to {save_path}")


def run_cnn(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    image_size: tuple[int, int] = (64, 64),
    epochs: int = 30,
):
    """Train and evaluate the CNN on the project dataset."""
    # gather file paths and labels, then build tf.data datasets with preprocessing
    train_paths, train_labels, classes = gather_image_paths(train_dir)
    val_paths, val_labels, _ = gather_image_paths(test_dir, classes)

    batch_size = 32
    train_ds = build_train_dataset(train_paths, train_labels, batch_size=batch_size, use_clahe=False)
    val_ds = build_val_dataset(val_paths, val_labels, batch_size=batch_size)
    train_eval_ds = build_val_dataset(train_paths, train_labels, batch_size=batch_size)
    save_dataset_examples(train_ds, classes, "train_preprocessed_examples.png")
    save_dataset_examples(val_ds, classes, "val_preprocessed_examples.png")

    model = CNNModel(
        input_shape=IMAGE_SIZE + (1,),
        num_classes=len(classes),
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=np.array(train_labels),
    )
    class_weights = dict(enumerate(class_weights_array))

    print("Class weights:", class_weights)

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping],
    )

    model.plot_history()
    test_loss, test_acc = model.evaluate(val_ds, verbose=2)

    model.plot_confusion_matrix(
        train_eval_ds,
        class_names=classes,
        save_path="train_confusion_matrix.png",
    )

    model.plot_confusion_matrix(
        val_ds,
        class_names=classes,
        save_path="val_confusion_matrix.png",
    )

    print("Classes:", classes)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Train label counts:", Counter(train_labels))
    print("Val label counts:", Counter(val_labels))
    print("Majority val baseline:", max(Counter(val_labels).values()) / len(val_labels))
    return model, history, (test_loss, test_acc)

'''
def run_svm(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
):
    """runs the SVM pipeline"""
    X_train, y_train, X_test, y_test, classes = load_dataset(
        train_dir=train_dir,
        test_dir=test_dir,
    )

    X_train, X_test = extract_features(X_train, X_test)
    X_train, X_test, _ = scale_features(X_train, X_test)
    svm = train_svm(X_train, y_train)
    evaluate_model(svm, X_test, y_test, classes)
    show_hog_example(TRAIN_DIR / "1_cumulus/1_cumulus_000009.jpg")
    return svm
'''


if __name__ == '__main__':
    #run_svm()
    run_cnn()
