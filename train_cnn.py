from pathlib import Path

import tensorflow as tf

from project_name.data.preprocessing import IMAGE_SIZE, load_and_preprocess_image
from project_name.models.cnn import CNNModel


PROJECT = Path(__file__).resolve().parent / "project_name"
DATASET = PROJECT / "data" / "dataset"
TRAIN_DIR = DATASET / "train"
TEST_DIR = DATASET / "test"


def get_classes(root):
    return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])


def gather_image_paths(root, classes=None):
    image_paths = []
    labels = []
    classes = classes or get_classes(root)

    for label, cls in enumerate(classes):
        class_dir = Path(root) / cls
        files = sorted(p for p in class_dir.iterdir() if p.is_file())

        for p in files:
            image_paths.append(str(p))
            labels.append(label)

        print(f"{cls}: {len(files)} images")

    return image_paths, labels, classes


def build_dataset(image_paths, labels, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(image_paths),
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    train_paths, train_labels, classes = gather_image_paths(TRAIN_DIR)
    val_paths, val_labels, _ = gather_image_paths(TEST_DIR, classes)

    train_ds = build_dataset(train_paths, train_labels, shuffle=True)
    val_ds = build_dataset(val_paths, val_labels)

    model = CNNModel(
        input_shape=IMAGE_SIZE + (1,),
        num_classes=len(classes),
        learning_rate=1e-4,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=1,
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(val_ds, verbose=2)
    print("Classes:", classes)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    save_path = PROJECT / "models" / "cnn_model.keras"
    model.model.save(str(save_path))
    print(f"Saved CNN model to {save_path}")


if __name__ == "__main__":
    main()