import tensorflow as tf

from project_name.data.loading import load_dataset, split
from project_name.data.preprocessing import (
    IMAGE_SIZE,
    grayify,
    normalise,
    reshape_for_cnn,
    resize,
)
from project_name.models.cnn import CNNModel


MODEL_PATH = "project_name/models/cnn_model.keras"
BATCH_SIZE = 32


def prepare_for_cnn(images):
    images = resize(images)
    images = grayify(images)
    images = normalise(images)
    images = reshape_for_cnn(images)
    return images


def make_dataset(images, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(labels),
            reshuffle_each_iteration=True,
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def run_cnn():
    X_train, y_train, X_test, y_test, classes = load_dataset()
    X_train, X_val, y_train, y_val = split(X_train, y_train)

    X_train = prepare_for_cnn(X_train)
    X_val = prepare_for_cnn(X_val)
    X_test = prepare_for_cnn(X_test)

    train_ds = make_dataset(X_train, y_train, shuffle=True)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

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

    model.summary()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        verbose=1,
        callbacks=[early_stopping],
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    model.plot_confusion_matrix(
        val_ds,
        class_names=classes,
        save_path="project_name/data/val_confusion_matrix.png",
    )

    model.model.save(MODEL_PATH)

    print("Classes:", classes)
    print("Validation loss:", val_loss)
    print("Validation accuracy:", val_acc)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print(f"Saved CNN model to {MODEL_PATH}")

    return model, history, (test_loss, test_acc)


if __name__ == "__main__":
    run_cnn()